"""ネットワーク自身をMCTSで動かして自己対戦し、AlphaZero方式の学習データ
（局面 → MCTSの訪問回数分布(方策ターゲット) → 最終的な勝敗）を収集する。
yonmoku_nn.selfplay（既存AIの模倣データ収集）とは別の、強化学習ループ用のスクリプト。
"""

import argparse
import json
from pathlib import Path

import torch

from .client import SimulationClient
from .encoding import SIZE
from .mcts import run_mcts, select_move
from .model import YonmokuNet

MAX_PLIES_PER_GAME = 300


def play_one_game(network, client: SimulationClient, num_simulations: int, c_puct: float,
                   temperature_plies: int) -> list[dict]:
    state = client.initial_state()
    samples = []
    ply = 0
    while not state["gameOver"] and ply < MAX_PLIES_PER_GAME:
        _, visit_counts = run_mcts(state, network, client, num_simulations, c_puct)
        temperature = 1.0 if ply < temperature_plies else 0.0
        move_idx = select_move(visit_counts, temperature)

        samples.append({
            "state_before": state,
            "visit_counts": {str(k): int(v) for k, v in visit_counts.items()},
            "mover": state["currentPlayer"],
        })

        r, c = move_idx // SIZE, move_idx % SIZE
        state = client.simulate_move(state, r, c)
        ply += 1

    winner = state["winner"] if state["gameOver"] else "draw"
    for sample in samples:
        sample["winner"] = winner
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="ネットワーク+MCTSによる自己対戦データ収集（AlphaZero方式）")
    parser.add_argument("--server", default="http://localhost:8080")
    parser.add_argument("--checkpoint", default="checkpoints/model.pt",
                         help="読み込む重み。存在しなければランダム初期化のネットワークから始める")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=100,
                         help="1手あたりのMCTSシミュレーション回数。多いほど強いが遅い")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temperature-plies", type=int, default=6,
                         help="この手数までは訪問回数に比例したサンプリング（多様性確保）、以降はほぼ決定的")
    parser.add_argument("--out", default="data/rl_selfplay.jsonl")
    args = parser.parse_args()

    network = YonmokuNet()
    if Path(args.checkpoint).exists():
        network.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"loaded checkpoint: {args.checkpoint}")
    else:
        print(f"checkpoint not found ({args.checkpoint}); starting from a randomly initialized network")
    network.eval()

    client = SimulationClient(args.server)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_positions = 0
    with out_path.open("a", encoding="utf-8") as f:
        for g in range(args.games):
            samples = play_one_game(network, client, args.simulations, args.c_puct, args.temperature_plies)
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            f.flush()
            total_positions += len(samples)
            winner = samples[-1]["winner"] if samples else None
            print(f"[{g + 1}/{args.games}] collected {len(samples)} positions "
                  f"(total {total_positions}), winner={winner}")

    print(f"done. wrote {total_positions} positions to {out_path}")


if __name__ == "__main__":
    main()
