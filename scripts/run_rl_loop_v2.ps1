<#
.SYNOPSIS
  RL loop v2: accumulate-then-fully-retrain, instead of the light per-generation fine-tune
  that run_rl_loop.ps1 does. Each cycle: (1) self-play a large batch of games from the
  current best checkpoint (written to its own file under -RlBufferDir, so every past
  cycle's games are kept), (2) retrain a fresh candidate from -FromCheckpoint using
  yonmoku_nn.train_combined -- which combines the ORIGINAL imitation-learning data
  (-ImitationDataGlob) with ALL accumulated self-play cycles so far, using the same
  robust recipe as the imitation training itself (many epochs, keep the epoch with the
  best val_loss) -- then (3) evaluate and gate like run_rl_loop.ps1 does.

  Why this exists: run_rl_loop.ps1's per-generation approach (train_rl.py, a light few-epoch
  fine-tune of the previous checkpoint using only that one generation's ~150 self-play games)
  reliably destroyed a strong imitation-learned checkpoint (model_base5.pt, ~35-50% vs
  DEFAULT) every single time it was tried, even at very low --lr/--epochs (1e-5 / 2 epochs).
  The diagnosis: a tiny, narrow self-play dataset (a hundredth the size of the 13,000-game
  imitation dataset) was overwriting carefully-converged weights in just a few gradient
  steps. This script avoids that by never doing a "light nudge" -- every retrain is a full,
  from-the-imitation-data-up training run, so a small new batch of self-play data can only
  gently shift the result, not overwrite it.

  Start the YonmokuRessen server in another window before running this (built from a
  checkout with the /api/simulate/ai-move endpoint, and NEURAL_MODEL_FILE pointed at
  checkpoints/model.onnx in this repo). Run this from the repository root
  (YonmokuRessen-Neural-Network).

.EXAMPLE
  # Start from the current best imitation-learned checkpoint, run 5 cycles of 1000
  # self-play games each, gated against DEFAULT.
  .\scripts\run_rl_loop_v2.ps1 -FromCheckpoint checkpoints/model_base5.pt -Cycles 5
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$FromCheckpoint,
    [int]$Cycles = 5,
    [int]$StartCycle = 1,
    # Glob pattern(s) for the original imitation-learning data (selfplay.py's output format --
    # actual moves played, not MCTS visit counts). Always included in every cycle's retrain.
    [string]$ImitationDataGlob = "data/selfplay_test3_v*.jsonl",
    # Where each cycle's self-play games are written (cycle1.jsonl, cycle2.jsonl, ...). All
    # files here are used together every cycle -- a genuinely growing replay buffer.
    [string]$RlBufferDir = "data/rl_buffer",
    [int]$SelfplayGames = 1000,
    [int]$SelfplaySimulations = 200,
    [int]$SelfplayConcurrency = 16,
    [double]$VsBuiltinRatio = 0.3,
    [string]$VsBuiltinLevels = "DEFAULT,TEST,TEST2,TEST3",
    # train_combined.py's own defaults already match the imitation recipe (epochs=40,
    # lr=1e-3) -- these let you override without editing the script.
    [int]$TrainEpochs = 40,
    [double]$Lr = 0.001,
    [int]$EvalGames = 30,
    [int]$EvalConcurrency = 6,
    [string]$Opponent = "DEFAULT",
    [string]$Server = "http://localhost:8080",
    [string]$PythonCpu = "..\.venv\Scripts\python.exe",
    [string]$PythonGpu = "..\.venv-rocm\Scripts\python.exe"
)

$logFile = "training_log_v2_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"

function Write-Log {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Host $line
    Add-Content -Path $logFile -Value $line -Encoding utf8
}

# Runs evaluate.py against a checkpoint and returns its win rate as a double (0-100),
# or -1 if the win rate could not be determined (evaluate.py failed or produced no
# parseable "win rate" line).
function Get-WinRate {
    param([string]$Checkpoint)

    $output = & $PythonCpu -m yonmoku_nn.evaluate --server $Server --opponent $Opponent --games $EvalGames `
        --random-opening-plies 4 --concurrency $EvalConcurrency --candidate $Checkpoint
    $exitCode = $LASTEXITCODE
    $output | ForEach-Object { Add-Content -Path $logFile -Value $_ -Encoding utf8 }

    if ($exitCode -ne 0) {
        Write-Log "WARNING: evaluate failed (exit $exitCode) while scoring $Checkpoint."
        return -1
    }
    $winRateLine = $output | Select-String "win rate:\s*([\d.]+)%" | Select-Object -First 1
    if (-not $winRateLine) {
        Write-Log "WARNING: could not find a win rate line while scoring $Checkpoint."
        return -1
    }
    return [double]$winRateLine.Matches[0].Groups[1].Value
}

Write-Log "=== RL loop v2 start: from $FromCheckpoint, ${Cycles} cycle(s) requested (starting at cycle $StartCycle), gated vs $Opponent ==="
Write-Log "selfplay: games=$SelfplayGames simulations=$SelfplaySimulations concurrency=$SelfplayConcurrency vs-builtin-ratio=$VsBuiltinRatio($VsBuiltinLevels) / train: imitation-data=$ImitationDataGlob lr=$Lr epochs=$TrainEpochs / eval: games=$EvalGames concurrency=$EvalConcurrency"

try {
    Invoke-WebRequest -Uri $Server -UseBasicParsing -TimeoutSec 5 | Out-Null
} catch {
    Write-Log "ERROR: cannot reach $Server. Start the YonmokuRessen server first. Aborting."
    exit 1
}

if (-not (Test-Path $FromCheckpoint)) {
    Write-Log "ERROR: $FromCheckpoint not found. Aborting."
    exit 1
}
New-Item -ItemType Directory -Force -Path $RlBufferDir | Out-Null

$bestCheckpoint = $FromCheckpoint

Write-Log "--- baseline: scoring the starting checkpoint ($bestCheckpoint) vs $Opponent ---"
$bestWinRate = Get-WinRate -Checkpoint $bestCheckpoint
if ($bestWinRate -lt 0) {
    Write-Log "ERROR: could not score the starting checkpoint. Aborting."
    exit 1
}
Write-Log "baseline win rate: $bestWinRate% ($bestCheckpoint)"

$cycle = $StartCycle - 1
for ($i = 1; $i -le $Cycles; $i++) {
    $cycle = $cycle + 1
    $selfplayOut = "$RlBufferDir/cycle${cycle}.jsonl"
    $candidateCheckpoint = "checkpoints/model_combined_cycle${cycle}.pt"

    Write-Log "--- cycle ${cycle}: self-play from $bestCheckpoint ($SelfplayGames games, simulations=$SelfplaySimulations, concurrency=$SelfplayConcurrency, vs-builtin-ratio=$VsBuiltinRatio) ---"
    & $PythonCpu -m yonmoku_nn.rl_selfplay --checkpoint $bestCheckpoint --games $SelfplayGames `
        --simulations $SelfplaySimulations --concurrency $SelfplayConcurrency --out $selfplayOut `
        --vs-builtin-ratio $VsBuiltinRatio --vs-builtin-levels $VsBuiltinLevels
    if ($LASTEXITCODE -ne 0) {
        Write-Log "ERROR: rl_selfplay failed at cycle ${cycle} (exit $LASTEXITCODE). Aborting."
        exit 1
    }

    Write-Log "--- cycle ${cycle}: combined retrain (GPU, imitation=$ImitationDataGlob, rl-buffer=$RlBufferDir/cycle*.jsonl, init=$bestCheckpoint, lr=$Lr, epochs=$TrainEpochs) ---"
    & $PythonGpu -m yonmoku_nn.train_combined --data $ImitationDataGlob "$RlBufferDir/cycle*.jsonl" `
        --init-checkpoint $bestCheckpoint --out $candidateCheckpoint --lr $Lr --epochs $TrainEpochs
    if ($LASTEXITCODE -ne 0) {
        Write-Log "ERROR: train_combined failed at cycle ${cycle} (exit $LASTEXITCODE). Aborting."
        exit 1
    }

    Write-Log "--- cycle ${cycle}: evaluate candidate (vs $Opponent, $EvalGames games) ---"
    $candidateWinRate = Get-WinRate -Checkpoint $candidateCheckpoint

    if ($candidateWinRate -lt 0) {
        Write-Log "cycle ${cycle} SKIPPED: could not evaluate $candidateCheckpoint (see WARNING above). Keeping $bestCheckpoint as the base for the next attempt."
    } elseif ($candidateWinRate -gt $bestWinRate) {
        Write-Log "cycle ${cycle} PROMOTED: $candidateWinRate% > previous best $bestWinRate%. New base: $candidateCheckpoint"
        $bestCheckpoint = $candidateCheckpoint
        $bestWinRate = $candidateWinRate
    } elseif ($candidateWinRate -eq $bestWinRate) {
        Write-Log "cycle ${cycle} TIED: $candidateWinRate% == previous best $bestWinRate%. Keeping $bestCheckpoint as the base for the next attempt (no strict improvement, so not promoted)."
    } else {
        Write-Log "cycle ${cycle} REJECTED: $candidateWinRate% < previous best $bestWinRate%. Keeping $bestCheckpoint as the base for the next attempt."
    }
}

Write-Log "=== RL loop v2 finished. Best checkpoint: $bestCheckpoint (win rate $bestWinRate% vs $Opponent) ==="
Write-Log "Full log is in this file: $logFile"
