"""YonmokuRessenサーバーにAI対AIの対局をいくつも作らせ、
「着手前の局面 → 実際に打たれた手 → 最終的な勝敗」の教師データをJSONL形式で収集する。

対局を最後まで進めるのはサーバー側のAiVsAiDriver（既存の仕組み）で、対局ごとに独立した
バックグラウンドスレッドが進行するため、複数の対局を並行して作成・待機できる
（--concurrencyで指定した数だけ同時に対局を進め、収集を大幅に高速化する）。
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .client import YonmokuClient


def _initial_state(size: int) -> dict:
    """1手目より前の初期状態。historyの1件目の「直前の状態」として使う。"""
    return {
        "size": size,
        "board": [[None] * size for _ in range(size)],
        "dmgMarks": [],
        "removalEchoes": {},
        "currentPlayer": "B",
        "hp": {"B": 6, "W": 6},
        "pending": None,
        "gameOver": False,
        "winner": None,
        "plyCount": 0,
        "lastMove": None,
    }


def collect_game(client: YonmokuClient, black_ai: str, white_ai: str,
                  poll_interval: float, timeout: float) -> list[dict]:
    game_id = client.create_game(black_ai, white_ai)
    history = client.wait_for_completion(game_id, poll_interval=poll_interval, timeout=timeout)
    if not history:
        return []

    size = history[0].get("size", 9)
    winner = history[-1]["winner"]  # "B" / "W" / "draw"

    samples = []
    prev_state = _initial_state(size)
    for state in history:
        move = state.get("lastMove")
        if move is not None:
            samples.append({
                "state_before": prev_state,
                "move": {"row": move["row"], "col": move["col"], "color": move["color"]},
                "winner": winner,
            })
        prev_state = state
    return samples


def main():
    parser = argparse.ArgumentParser(description="YonmokuRessenサーバーに自己対戦させ、学習データを収集する")
    parser.add_argument("--server", default="http://localhost:8080")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--black-ai", default="TEST3",
                         help="DEFAULT/TEST/TEST2/TEST3/LEARN")
    parser.add_argument("--white-ai", default="TEST3")
    parser.add_argument("--out", default="data/selfplay.jsonl")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=300.0,
                         help="1局あたりの最大待ち時間（秒）。HPがなかなか減らない長い対局は"
                              "まれに数分かかることがある")
    parser.add_argument("--concurrency", type=int, default=4,
                         help="同時に進行させる対局数。サーバー側は対局ごとに独立したスレッドで"
                              "進むため、増やすほど収集が速くなる（CPUコア数に応じて調整）")
    args = parser.parse_args()

    client = YonmokuClient(args.server)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_positions = 0
    completed = 0
    with out_path.open("a", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(collect_game, client, args.black_ai, args.white_ai,
                             args.poll_interval, args.timeout)
            for _ in range(args.games)
        ]
        failed = 0
        for future in as_completed(futures):
            completed += 1
            try:
                samples = future.result()
            except Exception as e:
                # 1局がタイムアウト/エラーになっても、他の対局の収集は続ける。
                failed += 1
                print(f"[{completed}/{args.games}] a game failed, skipping it: {e}")
                continue
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            f.flush()
            total_positions += len(samples)
            print(f"[{completed}/{args.games}] collected {len(samples)} positions (total {total_positions})")

    print(f"done. wrote {total_positions} positions to {out_path} ({failed} game(s) failed/skipped)")


if __name__ == "__main__":
    main()
