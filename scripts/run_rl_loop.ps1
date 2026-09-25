<#
.SYNOPSIS
  Runs the NEURAL AI reinforcement-learning loop (self-play -> train -> evaluate) for
  several generations in a row, unattended, logging progress and each generation's
  win rate against TEST3 to a timestamped log file.

  Start the YonmokuRessen server in another window before running this. For an
  unattended overnight run, set NEURAL_MCTS_SIMULATIONS back to something modest
  (e.g. 200-400) on the server so each evaluation game doesn't take too long.

  Run this from the repository root (YonmokuRessen-Neural-Network).

.EXAMPLE
  # checkpoints/model_rl_gen2.pt already exists; run 6 more generations (gen3..gen8)
  .\scripts\run_rl_loop.ps1 -FromGen 2 -Generations 6
#>

param(
    [Parameter(Mandatory = $true)]
    [int]$FromGen,
    [int]$Generations = 5,
    [int]$SelfplayGames = 60,
    [int]$SelfplaySimulations = 200,
    [int]$SelfplayConcurrency = 6,
    [int]$EvalGames = 10,
    [string]$Server = "http://localhost:8080",
    [string]$PythonCpu = "..\.venv\Scripts\python.exe",
    [string]$PythonGpu = "..\.venv-rocm\Scripts\python.exe",
    # Train on ALL accumulated self-play data (a growing replay buffer), not just the
    # newest generation's file. Training on only the latest ~60 games each time was
    # found to make win rate against TEST3 stay flat at 0% for 29 generations straight.
    [string]$DataGlob = "data/rl_selfplay_gen*.jsonl"
)

$logFile = "training_log_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"

function Write-Log {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Host $line
    Add-Content -Path $logFile -Value $line -Encoding utf8
}

Write-Log "=== RL loop start: from gen ${FromGen}, ${Generations} generation(s) requested ==="
Write-Log "selfplay: games=$SelfplayGames simulations=$SelfplaySimulations concurrency=$SelfplayConcurrency / eval: games=$EvalGames"

try {
    Invoke-WebRequest -Uri $Server -UseBasicParsing -TimeoutSec 5 | Out-Null
} catch {
    Write-Log "ERROR: cannot reach $Server. Start the YonmokuRessen server first. Aborting."
    exit 1
}

$prevGen = $FromGen
for ($i = 1; $i -le $Generations; $i++) {
    $gen = $prevGen + 1
    $prevCheckpoint = "checkpoints/model_rl_gen${prevGen}.pt"
    $selfplayOut = "data/rl_selfplay_gen${gen}.jsonl"
    $newCheckpoint = "checkpoints/model_rl_gen${gen}.pt"

    if (-not (Test-Path $prevCheckpoint)) {
        Write-Log "ERROR: $prevCheckpoint not found. Aborting."
        exit 1
    }

    Write-Log "--- gen ${gen}: self-play ($SelfplayGames games, simulations=$SelfplaySimulations, concurrency=$SelfplayConcurrency) ---"
    & $PythonCpu -m yonmoku_nn.rl_selfplay --checkpoint $prevCheckpoint --games $SelfplayGames `
        --simulations $SelfplaySimulations --concurrency $SelfplayConcurrency --out $selfplayOut
    if ($LASTEXITCODE -ne 0) {
        Write-Log "ERROR: rl_selfplay failed at gen ${gen} (exit $LASTEXITCODE). Aborting."
        exit 1
    }

    Write-Log "--- gen ${gen}: training (GPU, data=$DataGlob) ---"
    & $PythonGpu -m yonmoku_nn.train_rl --data $DataGlob --init-checkpoint $prevCheckpoint --out $newCheckpoint
    if ($LASTEXITCODE -ne 0) {
        Write-Log "ERROR: train_rl failed at gen ${gen} (exit $LASTEXITCODE). Aborting."
        exit 1
    }

    Write-Log "--- gen ${gen}: evaluate (vs TEST3, $EvalGames games) ---"
    # Do not merge stderr into this capture (no 2>&1): export.py prints a harmless
    # DeprecationWarning there, and PowerShell can otherwise treat captured native-command
    # stderr output as an error. Only stdout (the actual progress/result lines) is needed.
    $evalOutput = & $PythonCpu -m yonmoku_nn.evaluate --server $Server --opponent TEST3 --games $EvalGames `
        --random-opening-plies 4 --concurrency 4 --candidate $newCheckpoint
    $evalExitCode = $LASTEXITCODE
    $evalOutput | ForEach-Object { Add-Content -Path $logFile -Value $_ -Encoding utf8 }
    if ($evalExitCode -ne 0) {
        Write-Log "WARNING: evaluate failed at gen ${gen} (exit $evalExitCode). Checkpoint was still saved; continuing to the next generation."
    } else {
        $winRateLine = $evalOutput | Select-String "win rate"
        Write-Log "gen ${gen} result: $winRateLine"
    }

    $prevGen = $gen
}

Write-Log "=== RL loop finished. Latest checkpoint: checkpoints/model_rl_gen${prevGen}.pt ==="
Write-Log "Full log is in this file: $logFile"
