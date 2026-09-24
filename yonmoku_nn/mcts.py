"""PUCT（AlphaZero系）モンテカルロ木探索。ネットワークの方策で子ノードの事前確率を決め、
価値ヘッドで葉ノードを評価する（ランダムロールアウトは行わない）。
新しい局面（まだ木に無い子）が必要になった時だけ、YonmokuRessenサーバーのステートレスな
シミュレーションAPI（SimulationClient）を呼んでルール適用結果を取得する。盤面ルール自体は
一切ここで再実装しない。
"""

import math

import numpy as np
import torch

from .client import SimulationClient
from .encoding import SIZE, encode_state


def legal_moves(state: dict) -> list[int]:
    board = state["board"]
    moves = []
    for r in range(len(board)):
        for c in range(len(board[r])):
            if board[r][c] is None:
                moves.append(r * SIZE + c)
    return moves


def terminal_value(state: dict) -> float:
    """終局状態の評価値。stateのcurrentPlayerは「終局させた側（直前に着手した側）」を指す
    （GameRoom.applyTurnの仕様：gameOver時はcurrentPlayerを次の手番へ進めない）。"""
    winner = state.get("winner")
    if winner is None or winner == "draw":
        return 0.0
    return 1.0 if winner == state["currentPlayer"] else -1.0


class MCTSNode:
    __slots__ = ("state", "prior", "children", "priors", "visit_count", "value_sum", "expanded")

    def __init__(self, state: dict, prior: float = 0.0):
        self.state = state
        self.prior = prior
        self.children: dict[int, "MCTSNode"] = {}
        self.priors: dict[int, float] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.expanded = False

    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


def _expand(node: MCTSNode, network) -> float:
    """未展開のノードに対し、ネットワークで方策・価値を計算する。子ノードの実際の状態は
    ここでは作らない（選ばれた時点でシミュレーションAPIを呼んで初めて作る＝遅延展開）。
    戻り値はこのノード（node.stateのcurrentPlayer視点）の価値評価。
    """
    moves = legal_moves(node.state)
    x = encode_state(node.state, node.state["currentPlayer"])
    x_tensor = torch.from_numpy(x).unsqueeze(0)
    with torch.no_grad():
        policy_logits, value = network(x_tensor)
    policy = torch.softmax(policy_logits[0], dim=0).numpy()

    total = 0.0
    for move_idx in moves:
        p = float(policy[move_idx])
        node.priors[move_idx] = p
        total += p
    if total > 1e-8:
        for k in node.priors:
            node.priors[k] /= total
    else:
        # ネットワークの出力がすべて0に近い異常系のフォールバック（一様分布）。
        uniform = 1.0 / len(moves) if moves else 0.0
        for k in moves:
            node.priors[k] = uniform

    node.expanded = True
    return float(value[0].item())


def _select_move(node: MCTSNode, c_puct: float) -> int:
    sqrt_total = math.sqrt(sum(child.visit_count for child in node.children.values()) + 1)
    best_score = -float("inf")
    best_move = None
    for move_idx, prior in node.priors.items():
        child = node.children.get(move_idx)
        q = child.value() if child is not None else 0.0
        n = child.visit_count if child is not None else 0
        u = c_puct * prior * sqrt_total / (1 + n)
        score = q + u
        if score > best_score:
            best_score = score
            best_move = move_idx
    return best_move


def _simulate_once(root: MCTSNode, network, client: SimulationClient, c_puct: float) -> None:
    path = [root]
    node = root

    # 1. 既知の木を、既存の子だけを辿ってPUCTで下る（新しい子が必要になったら止まる）。
    move_idx = None
    while node.expanded and not node.state["gameOver"]:
        move_idx = _select_move(node, c_puct)
        child = node.children.get(move_idx)
        if child is None:
            break
        node = child
        path.append(node)
        move_idx = None

    # 2. 葉に到達した場合の評価。
    if node.state["gameOver"]:
        value = terminal_value(node.state)
    elif not node.expanded:
        value = _expand(node, network)
    else:
        # 選ばれた手の子がまだ無い＝ここでシミュレーションAPIを呼んで新規作成する。
        r, c = move_idx // SIZE, move_idx % SIZE
        new_state = client.simulate_move(node.state, r, c)
        child = MCTSNode(new_state, prior=node.priors[move_idx])
        node.children[move_idx] = child
        path.append(child)
        node = child
        if node.state["gameOver"]:
            value = terminal_value(node.state)
        else:
            value = _expand(node, network)

    # 3. バックプロパゲーション。手番は1階層ごとに入れ替わるため、価値の符号も反転させながら遡る。
    for n in reversed(path):
        n.visit_count += 1
        n.value_sum += value
        value = -value


def run_mcts(root_state: dict, network, client: SimulationClient, num_simulations: int,
             c_puct: float = 1.5) -> tuple[MCTSNode, dict[int, int]]:
    """root_stateから num_simulations 回の探索を行い、ルートノードと各手の訪問回数を返す。"""
    root = MCTSNode(root_state)
    value = _expand(root, network)
    root.visit_count = 1
    root.value_sum = value

    for _ in range(num_simulations):
        _simulate_once(root, network, client, c_puct)

    visit_counts = {
        move_idx: (root.children[move_idx].visit_count if move_idx in root.children else 0)
        for move_idx in root.priors
    }
    return root, visit_counts


def select_move(visit_counts: dict[int, int], temperature: float) -> int:
    """訪問回数の分布から実際に打つ手を選ぶ。temperature<=0でほぼ決定的（最頻訪問手）、
    temperature=1.0で訪問回数に比例したサンプリング（自己対戦での多様性確保用）。"""
    moves = list(visit_counts.keys())
    counts = np.array([visit_counts[m] for m in moves], dtype=np.float64)
    if temperature <= 1e-3:
        idx = int(np.argmax(counts))
    else:
        powered = counts ** (1.0 / temperature)
        total = powered.sum()
        if total <= 1e-8:
            idx = int(np.argmax(counts))
        else:
            probs = powered / total
            idx = int(np.random.choice(len(moves), p=probs))
    return moves[idx]
