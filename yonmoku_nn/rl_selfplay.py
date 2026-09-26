"""ネットワーク自身をMCTSで動かして自己対戦し、AlphaZero方式の学習データ
（局面 → MCTSの訪問回数分布(方策ターゲット) → 最終的な勝敗）を収集する。
yonmoku_nn.selfplay（既存AIの模倣データ収集）とは別の、強化学習ループ用のスクリプト。

selfplay.py（既存AIの模倣データ収集）は対局の進行自体がサーバー側スレッドで行われるため
ThreadPoolExecutorだけで十分並列化できるが、こちらはMCTS探索そのものがPython側のCPU処理
（ネットワーク推論）なので、GILの制約を受けないmultiprocessingのワーカープロセスで並列化する
（--concurrencyで指定した数だけ対局を分担する）。
"""

import argparse
import json
import multiprocessing as mp
import random
from pathlib import Path

import torch

from .client import SimulationClient
from .encoding import SIZE
from .mcts import run_mcts, select_move
from .model import YonmokuNet

MAX_PLIES_PER_GAME = 300


def play_one_game(network, client: SimulationClient, num_simulations: int, c_puct: float,
                   temperature_plies: int, opponent_level: str | None = None,
                   network_color: str = "B") -> list[dict]:
    """1局分の自己対戦データを集める。

    opponent_levelがNoneなら、これまで通り両者ともネットワーク+MCTSで打つ（純粋な自己対戦）。
    指定すると、network_color側だけがネットワーク+MCTSで打ち、もう片方は内蔵AI（DEFAULT等、
    /api/simulate/ai-move）が打つ。これは、ネットワーク同士の対戦だけを続けると、内蔵AIのような
    自分とは異なる打ち方への対応力を失っていく（自己対戦の戦略崩壊）ことへの対策で、学習データ
    （MCTSの訪問回数分布）はnetwork_color側の手番でのみ記録する（内蔵AI側の手はそのまま適用する
    だけで、方策の教師データとしては使わない）。
    """
    state = client.initial_state()
    samples = []
    ply = 0
    while not state["gameOver"] and ply < MAX_PLIES_PER_GAME:
        is_network_turn = opponent_level is None or state["currentPlayer"] == network_color
        if is_network_turn:
            _, visit_counts = run_mcts(state, network, client, num_simulations, c_puct)
            temperature = 1.0 if ply < temperature_plies else 0.0
            move_idx = select_move(visit_counts, temperature)

            samples.append({
                "state_before": state,
                "visit_counts": {str(k): int(v) for k, v in visit_counts.items()},
                "mover": state["currentPlayer"],
            })

            r, c = move_idx // SIZE, move_idx % SIZE
        else:
            move = client.ai_move(state, opponent_level)
            if move is None:
                break
            r, c = move

        state = client.simulate_move(state, r, c)
        ply += 1

    winner = state["winner"] if state["gameOver"] else "draw"
    for sample in samples:
        sample["winner"] = winner
    return samples


def _load_network(checkpoint: str) -> YonmokuNet:
    network = YonmokuNet()
    if Path(checkpoint).exists():
        network.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    network.eval()
    return network


def _run_worker(worker_id: int, server: str, checkpoint: str, games: int, num_simulations: int,
                 c_puct: float, temperature_plies: int, out_path: str,
                 vs_builtin_ratio: float = 0.0, vs_builtin_levels: tuple[str, ...] = ()) -> int:
    """1ワーカープロセス分の自己対戦を行い、書き出した局面数を返す。
    ワーカーごとに別プロセスなので、ネットワークの重みとHTTPセッションもそれぞれ独立して持つ。
    torch側の内部スレッド並列化は、プロセス並列と二重に競合しないよう1に絞る。

    vs_builtin_ratio > 0 なら、そのうち一部の局はネットワーク同士ではなく、片方を
    vs_builtin_levelsからランダムに選んだ内蔵AIにして対戦する（自己対戦の戦略崩壊対策）。
    """
    torch.set_num_threads(1)
    network = _load_network(checkpoint)
    client = SimulationClient(server)

    total = 0
    failed = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for g in range(games):
            opponent_level = None
            network_color = "B"
            if vs_builtin_ratio > 0 and vs_builtin_levels and random.random() < vs_builtin_ratio:
                opponent_level = random.choice(vs_builtin_levels)
                network_color = random.choice(["B", "W"])
            try:
                samples = play_one_game(network, client, num_simulations, c_puct, temperature_plies,
                                         opponent_level=opponent_level, network_color=network_color)
            except Exception as e:
                # 1局がネットワークの瞬断などで失敗しても、このワーカーの他の対局・
                # 他のワーカーがそれまでに集めたデータは失わずに続行する。
                failed += 1
                print(f"[worker {worker_id}] [{g + 1}/{games}] a game failed, skipping it: {e}")
                continue
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            f.flush()
            total += len(samples)
            winner = samples[-1]["winner"] if samples else None
            opponent_desc = f"vs {opponent_level}(network={network_color})" if opponent_level else "self-play"
            print(f"[worker {worker_id}] [{g + 1}/{games}] collected {len(samples)} positions "
                  f"(total {total}), winner={winner}, {opponent_desc}")
    if failed:
        print(f"[worker {worker_id}] done. {failed} game(s) failed/skipped.")
    return total


def _distribute(games: int, workers: int) -> list[int]:
    """対局数をワーカーになるべく均等に割り振る（余りは先頭のワーカーから1局ずつ多く持たせる）。"""
    base, extra = divmod(games, workers)
    return [base + (1 if i < extra else 0) for i in range(workers)]


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
    parser.add_argument("--concurrency", type=int, default=1,
                         help="対局を分担して並列に自己対戦するワーカープロセス数。MCTS自体がCPU処理"
                              "（ネットワーク推論）なので、物理コア数程度まで増やすと収集時間を短縮できる")
    parser.add_argument("--vs-builtin-ratio", type=float, default=0.0,
                         help="このうち一部の局を、ネットワーク同士ではなく片方を内蔵AIにして対戦させる"
                              "割合（0〜1）。純粋な自己対戦だけを続けると、内蔵AIのような自分とは違う"
                              "打ち方への対応力を失っていく（自己対戦の戦略崩壊）ことへの対策")
    parser.add_argument("--vs-builtin-levels", default="DEFAULT,TEST,TEST2,TEST3",
                         help="--vs-builtin-ratioで内蔵AI戦になった場合、この中からランダムに1つ選ぶ"
                              "（カンマ区切り）")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"checkpoint not found ({args.checkpoint}); starting from a randomly initialized network")
    else:
        print(f"loaded checkpoint: {args.checkpoint}")

    vs_builtin_levels = tuple(level.strip() for level in args.vs_builtin_levels.split(",") if level.strip())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.concurrency <= 1:
        total_positions = _run_worker(0, args.server, args.checkpoint, args.games, args.simulations,
                                       args.c_puct, args.temperature_plies, str(out_path),
                                       args.vs_builtin_ratio, vs_builtin_levels)
        print(f"done. wrote {total_positions} positions to {out_path}")
        return

    games_per_worker = [n for n in _distribute(args.games, args.concurrency) if n > 0]
    tmp_paths = [out_path.with_name(f"{out_path.stem}.part{i}{out_path.suffix}")
                 for i in range(len(games_per_worker))]
    worker_args = [
        (i, args.server, args.checkpoint, n, args.simulations, args.c_puct, args.temperature_plies, str(p),
         args.vs_builtin_ratio, vs_builtin_levels)
        for i, (n, p) in enumerate(zip(games_per_worker, tmp_paths))
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(worker_args)) as pool:
        results = pool.starmap(_run_worker, worker_args)
    total_positions = sum(results)

    with out_path.open("a", encoding="utf-8") as out_f:
        for p in tmp_paths:
            with p.open("r", encoding="utf-8") as in_f:
                out_f.write(in_f.read())
            p.unlink()

    print(f"done. wrote {total_positions} positions to {out_path} "
          f"(using {len(worker_args)} worker process(es))")


if __name__ == "__main__":
    main()
