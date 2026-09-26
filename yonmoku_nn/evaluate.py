"""NEURAL AIの強さを、既存の内蔵AI（既定でTEST3）との対局を通じて客観的に測る。

強化学習ループ（rl_selfplay → train_rl → export）を1世代回すたびに「本当に強くなったか」を
確認せずに次の世代へ進むと、弱くなっていることに気づけない。対局は実際にサーバーへ
（AiVsAiDriverで）作らせるので、対局時の挙動（NeuralMcts含む）をそのまま評価できる。

--candidateを指定すると、その.ptチェックポイントを一時的にエクスポートしてサーバーの
NEURAL_MODEL_FILEに反映してから評価する（サーバーの自動リロードが検知するまで少し待つ）。
省略した場合は、サーバーに現在読み込まれているモデルをそのまま評価する。

NEURAL（NeuralMcts）もTEST3（アルファベータ）も着手選択が完全に決定論的（乱数を使わない）
なため、何も工夫しないと「NEURALが先手」「NEURALが後手」の2パターンしか実質的な対局が
存在せず、--gamesを増やしても同じ対局を繰り返すだけになる。それを避けるため、対局ごとに
最初の数手（--random-opening-plies）だけランダムな手を人間役として打ってから、両者に
AIを割り当てて残りを進めさせる（サーバー側の新しいAPI: POST /api/games/{id}/ai）。

対局の進行自体はサーバー側のAiVsAiDriverが独立したスレッドで行うため、selfplay.pyと同様に
ThreadPoolExecutorで複数局を並行して作成・待機できる（--concurrency）。rl_selfplay.pyの
マルチプロセス化とは違い、Python側はただ作って待つだけの軽い処理なのでスレッドで十分。
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .client import YonmokuClient


def _play_one_game(client: YonmokuClient, opponent: str, random_opening_plies: int,
                    poll_interval: float, timeout: float, neural_color: str) -> str:
    black_ai, white_ai = ("NEURAL", opponent) if neural_color == "B" else (opponent, "NEURAL")
    game_id = client.create_game()
    client.place_random_opening_moves(game_id, random_opening_plies)
    client.assign_ai(game_id, black_ai, white_ai)
    history = client.wait_for_completion(game_id, poll_interval=poll_interval, timeout=timeout)
    return history[-1]["winner"]


def play_match(client: YonmokuClient, games: int, opponent: str, random_opening_plies: int,
               poll_interval: float, timeout: float, concurrency: int) -> dict:
    wins = losses = draws = failed = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # 先後を交互にして、色による有利不利（先手番の方が有利、等）を打ち消す。
        futures = {
            executor.submit(_play_one_game, client, opponent, random_opening_plies,
                             poll_interval, timeout, "B" if g % 2 == 0 else "W"): ("B" if g % 2 == 0 else "W")
            for g in range(games)
        }
        for future in as_completed(futures):
            neural_color = futures[future]
            completed += 1
            try:
                winner = future.result()
            except Exception as e:
                failed += 1
                print(f"[{completed}/{games}] a game failed, skipping it: {e}")
                continue

            if winner == neural_color:
                wins += 1
            elif winner is None or winner == "draw":
                draws += 1
            else:
                losses += 1
            print(f"[{completed}/{games}] NEURAL={neural_color} vs {opponent} -> winner={winner}")

    return {"wins": wins, "losses": losses, "draws": draws, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description="NEURAL AIの強さを既存AIとの対局で評価する")
    parser.add_argument("--server", default="http://localhost:8080")
    parser.add_argument("--opponent", default="TEST3", help="DEFAULT/TEST/TEST2/TEST3/LEARN")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--random-opening-plies", type=int, default=4,
                         help="対局ごとに、AIを割り当てる前にランダムな手を何手打たせておくか。"
                              "NEURAL・TEST3とも決定論的なので、0のままだと--gamesを増やしても"
                              "先手・後手2パターンの繰り返しにしかならない")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=300.0,
                         help="1局あたりの最大待ち時間（秒）。NEURALはMCTS探索のぶん1手ごとに"
                              "時間がかかるので、内蔵AI同士より長めにしてある")
    parser.add_argument("--concurrency", type=int, default=4,
                         help="同時に進行させる対局数。対局はサーバー側のスレッドで進むため、"
                              "増やすほど評価が速くなる（サーバーのCPUコア数に応じて調整）")
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
    result = play_match(client, args.games, args.opponent, args.random_opening_plies,
                         args.poll_interval, args.timeout, args.concurrency)

    total = result["wins"] + result["losses"] + result["draws"]
    print()
    print(f"vs {args.opponent}: {result['wins']}勝 {result['losses']}敗 {result['draws']}分け"
          f"（{result['failed']}局失敗/スキップ）")

    if total == 0:
        # 1局も完了しなかった場合、win rate: 0.0%と表示してしまうと「本当に0%だった」のか
        # 「評価自体が全滅した」のか区別が付かず、呼び出し元（run_rl_loop.ps1など）が
        # 昇格判定でこれを実際の0%として扱ってしまう(=評価失敗を検知できない)。
        # そのため0局は明確に失敗として扱う。
        print(f"ERROR: all {result['failed']} game(s) failed; could not measure a win rate.",
              file=sys.stderr)
        sys.exit(1)

    win_rate = result["wins"] / total
    print(f"win rate: {win_rate:.1%} ({total} games)")


if __name__ == "__main__":
    main()
