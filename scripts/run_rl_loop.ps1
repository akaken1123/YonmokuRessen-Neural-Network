<#
.SYNOPSIS
  NEURAL AIを複数世代にわたって自動的に強化学習するループ。
  自己対戦(rl_selfplay, CPU) → 学習(train_rl, GPU) → 評価(evaluate) を1世代分として、
  指定した世代数だけ繰り返し、結果をログファイル（training_log_日時.txt）に記録する。

  実行前に別ウィンドウでYonmokuRessenサーバーを起動しておくこと。
  対人戦用にNEURAL_MCTS_SIMULATIONSは200〜400程度に戻しておくのがおすすめ
  （評価のたびに時間がかかりすぎるのを防ぐため）。

  リポジトリのルート（YonmokuRessen-Neural-Network）から実行すること。

.EXAMPLE
  # 今 checkpoints/model_rl_gen2.pt まであるとして、そこから6世代（gen3〜gen8）を回す
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
    [string]$PythonGpu = "..\.venv-rocm\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$logFile = "training_log_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"

function Write-Log {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

Write-Log "=== RL loop start: from gen$FromGen, $Generations generation(s) requested ==="
Write-Log "selfplay: games=$SelfplayGames simulations=$SelfplaySimulations concurrency=$SelfplayConcurrency / eval: games=$EvalGames"

try {
    Invoke-WebRequest -Uri $Server -UseBasicParsing -TimeoutSec 5 | Out-Null
} catch {
    Write-Log "ERROR: $Server に接続できません。先にYonmokuRessenサーバーを起動してください。中断します。"
    exit 1
}

$prevGen = $FromGen
for ($i = 1; $i -le $Generations; $i++) {
    $gen = $prevGen + 1
    $prevCheckpoint = "checkpoints/model_rl_gen$prevGen.pt"
    $selfplayOut = "data/rl_selfplay_gen$gen.jsonl"
    $newCheckpoint = "checkpoints/model_rl_gen$gen.pt"

    if (-not (Test-Path $prevCheckpoint)) {
        Write-Log "ERROR: $prevCheckpoint が見つかりません。中断します。"
        exit 1
    }

    Write-Log "--- gen$gen: 自己対戦 ($SelfplayGames局, simulations=$SelfplaySimulations, concurrency=$SelfplayConcurrency) ---"
    & $PythonCpu -m yonmoku_nn.rl_selfplay --checkpoint $prevCheckpoint --games $SelfplayGames `
        --simulations $SelfplaySimulations --concurrency $SelfplayConcurrency --out $selfplayOut
    if ($LASTEXITCODE -ne 0) {
        Write-Log "ERROR: rl_selfplay が gen$gen で失敗しました (exit $LASTEXITCODE)。中断します。"
        exit 1
    }

    Write-Log "--- gen$gen: 学習 (GPU) ---"
    & $PythonGpu -m yonmoku_nn.train_rl --data $selfplayOut --init-checkpoint $prevCheckpoint --out $newCheckpoint
    if ($LASTEXITCODE -ne 0) {
        Write-Log "ERROR: train_rl が gen$gen で失敗しました (exit $LASTEXITCODE)。中断します。"
        exit 1
    }

    Write-Log "--- gen$gen: 評価 (vs TEST3, $EvalGames局) ---"
    $evalOutput = & $PythonCpu -m yonmoku_nn.evaluate --server $Server --opponent TEST3 --games $EvalGames `
        --random-opening-plies 4 --concurrency 4 --candidate $newCheckpoint 2>&1
    $evalOutput | ForEach-Object { Add-Content -Path $logFile -Value $_ }
    $winRateLine = $evalOutput | Select-String "win rate"
    Write-Log "gen$gen 結果: $winRateLine"

    $prevGen = $gen
}

Write-Log "=== RL loop finished. 最新チェックポイント: checkpoints/model_rl_gen$prevGen.pt ==="
Write-Log "ログ全文はこのファイルにあります: $logFile"
