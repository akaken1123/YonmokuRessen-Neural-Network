"""YonmokuRessenサーバーが返すGameStateSnapshot(JSON)を、
ニューラルネット入力用のテンソル(NUM_PLANES, SIZE, SIZE)へ変換する。

盤面ルール自体はJava側（GameRoom）が唯一の実装であり、ここでは何も再現しない。
サーバーが返した状態をそのまま数値表現に変換するだけ。
"""

import numpy as np

SIZE = 9
NUM_PLANES = 13

# 各チャンネルの意味はREADME.mdの表を参照。
PLANE_OWN_STONE = 0
PLANE_OWN_DMG_FLAG = 1
PLANE_OPP_STONE = 2
PLANE_OPP_DMG_FLAG = 3
PLANE_OWN_BACK_ATTACK = 4
PLANE_OPP_BACK_ATTACK = 5
PLANE_DMG_MARK = 6
PLANE_ECHO_MARK = 7
PLANE_ECHO_MARK_WAS_DMG = 8
PLANE_OWN_HP = 9
PLANE_OPP_HP = 10
PLANE_PENDING_AGAINST_ME = 11
PLANE_PENDING_AGAINST_OPPONENT = 12

_MAX_HP = 6.0
_PENDING_NORMALIZER = 10.0


def _key(r: int, c: int) -> str:
    return f"{r},{c}"


def encode_state(state: dict, perspective: str) -> np.ndarray:
    """
    state: GameStateSnapshotをそのままJSONデコードしたdict。
    perspective: この局面で着手する側の色（"B"または"W"）。
                 常にperspective側が「自分」チャンネルになるよう正規化する。
    """
    opponent = "W" if perspective == "B" else "B"
    board = state["board"]
    dmg_marks = set(state.get("dmgMarks") or [])
    removal_echoes = state.get("removalEchoes") or {}
    hp = state.get("hp") or {}
    pending = state.get("pending")

    planes = np.zeros((NUM_PLANES, SIZE, SIZE), dtype=np.float32)

    for r in range(SIZE):
        row = board[r]
        for c in range(SIZE):
            cell = row[c]
            if cell is not None:
                color = cell["color"]
                is_own = color == perspective
                planes[PLANE_OWN_STONE if is_own else PLANE_OPP_STONE, r, c] = 1.0
                if cell.get("dmgFlag"):
                    planes[PLANE_OWN_DMG_FLAG if is_own else PLANE_OPP_DMG_FLAG, r, c] = 1.0
                if cell.get("backAttackBonus", 0) > 0:
                    planes[PLANE_OWN_BACK_ATTACK if is_own else PLANE_OPP_BACK_ATTACK, r, c] = 1.0
            else:
                k = _key(r, c)
                if k in dmg_marks:
                    planes[PLANE_DMG_MARK, r, c] = 1.0
                if k in removal_echoes:
                    planes[PLANE_ECHO_MARK, r, c] = 1.0
                    if removal_echoes[k]:
                        planes[PLANE_ECHO_MARK_WAS_DMG, r, c] = 1.0

    planes[PLANE_OWN_HP, :, :] = hp.get(perspective, _MAX_HP) / _MAX_HP
    planes[PLANE_OPP_HP, :, :] = hp.get(opponent, _MAX_HP) / _MAX_HP

    if pending is not None:
        amount = pending.get("amount", 0) / _PENDING_NORMALIZER
        if pending.get("target") == perspective:
            planes[PLANE_PENDING_AGAINST_ME, :, :] = amount
        else:
            planes[PLANE_PENDING_AGAINST_OPPONENT, :, :] = amount

    return planes
