"""NEURAL AIの強さを、既存の内蔵AI（既定でTEST3）との対局を通じて客観的に測る。

強化学習ループ（rl_selfplay → train_rl → export）を1世代回すたびに「本当に強くなったか」を
確認せずに次の世代へ進むと、弱くなっていることに気づけない。対局は実際にサーバーへ
（AiVsAiDriverで）作らせるので、対局時の挙動（NeuralMcts含む）をそのまま評価できる。

--candidateを指定すると、その.ptチェックポイントを一時的にエクスポートしてサーバーの
NEURAL_MODEL_FILEに反映してから評価する（サーバーの自動リロードが検知するまで少し待つ）。
省略した場合は、サーバーに現在読み込まれているモデルをそのまま評価する。
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from .client import YonmokuClient


def play_match(client: YonmokuClient, games: int, opponent: str,
                poll_interval: float, timeout: float) -> dict:
    wins = losses = draws = failed = 0
    for g in range(games):
        # 先後を交互にして、色による有利不利（先手番の方が有利、等）を打ち消す。
        if g % 2 == 0:
            black_ai, white_ai, neural_color = "NEURAL", opponent, "B"
        else:
            black_ai, white_ai, neural_color = opponent, "NEURAL", "W"

        try:
            game_id = client.create_game(black_ai, white_ai)
            history = client.wait_for_completion(game_id, poll_interval=poll_interval, timeout=timeout)
        except Exception as e:
            failed += 1
            print(f"[{g + 1}/{games}] a game failed, skipping it: {e}")
            continue

        winner = history[-1]["winner"]
        if winner == neural_color:
            wins += 1
        elif winner is None or winner == "draw":
            draws += 1
        else:
            losses += 1
        print(f"[{g + 1}/{games}] NEURAL={neural_color} vs {opponent} -> winner={winner}")

    return {"wins": wins, "losses": losses, "draws": draws, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description="NEURAL AIの強さを既存AIとの対局で評価する")
    parser.add_argument("--server", default="http://localhost:8080")
    parser.add_argument("--opponent", default="TEST3", help="DEFAULT/TEST/TEST2/TEST3/LEARN")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=300.0,
                         help="1局あたりの最大待ち時間（秒）。NEURALはMCTS探索のぶん1手ごとに"
                              "時間がかかるので、内蔵AI同士より長めにしてある")
    parser.add_argument("--candidate", default=None,
                         help="評価したい.ptチェックポイント。省略時はサーバーに現在読み込まれている"
                              "モデルをそのまま評価する")
    parser.add_argument("--model-file", default="checkpoints/model.onnx",
                         help="サーバーのNEURAL_MODEL_FILEが指しているのと同じパス"
                              "（--candidate指定時、ここへエクスポートする）")
    parser.add_argument("--wait-seconds", type=float, default=35.0,
                         help="モデルファイル差し替え後、サーバーの自動リロード"
                              "（既定30秒間隔）が検知するのを待つ秒数")
    args = parser.parse_args()

    if args.candidate:
        out_path = Path(args.model_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"exporting {args.candidate} -> {args.model_file}")
        subprocess.run(
            [sys.executable, "-m", "yonmoku_nn.export", "--checkpoint", args.candidate, "--out", args.model_file],
            check=True,
        )
        print(f"waiting {args.wait_seconds}s for the server to auto-reload the new model...")
        time.sleep(args.wait_seconds)

    client = YonmokuClient(args.server)
    result = play_match(client, args.games, args.opponent, args.poll_interval, args.timeout)

    total = result["wins"] + result["losses"] + result["draws"]
    win_rate = result["wins"] / total if total else 0.0
    print()
    print(f"vs {args.opponent}: {result['wins']}勝 {result['losses']}敗 {result['draws']}分け"
          f"（{result['failed']}局失敗/スキップ）")
    print(f"win rate: {win_rate:.1%} ({total} games)")


if __name__ == "__main__":
    main()
