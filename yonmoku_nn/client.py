"""起動中のYonmokuRessen Javaサーバーと通信する薄いREST APIクライアント。
盤面ルールはサーバー側（Java）の実装が唯一の正解であり、ここでは一切再実装しない。
"""

import random
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class YonmokuClient:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")

    def create_game(self, black_ai: str | None = None, white_ai: str | None = None) -> str:
        """blackAi/whiteAiを両方指定すると、サーバー側のAiVsAiDriverが自動で最後まで対局を進める。
        両方省略すると、両者とも人間操作（place_moveで直接着手できる）の対局になる。"""
        params = {}
        if black_ai:
            params["blackAi"] = black_ai
        if white_ai:
            params["whiteAi"] = white_ai
        resp = requests.post(f"{self.base_url}/api/games", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()["id"]

    def get_state(self, game_id: str) -> dict:
        resp = requests.get(f"{self.base_url}/api/games/{game_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def place_move(self, game_id: str, row: int, col: int) -> dict:
        """人間操作中の色として着手する（NNやTEST3のようなAI対AI対局にはまだ切り替えていない対局のみ）。"""
        resp = requests.post(f"{self.base_url}/api/games/{game_id}/move", json={"row": row, "col": col}, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def place_random_opening_moves(self, game_id: str, plies: int) -> dict:
        """まだAIを割り当てていない対局に、ランダムな開始局面を作るため何手か適当に打つ。
        NEURAL/TEST3のどちらも着手選択が決定論的（乱数を使わない）なため、同じ対局カードを
        何度評価しても常に同じ結果になってしまう問題を避けるために使う（evaluate.py）。
        対局が終わってしまった場合はそこで打ち切る。"""
        state = self.get_state(game_id)
        for _ in range(plies):
            if state["gameOver"]:
                break
            board = state["board"]
            size = len(board)
            empties = [(r, c) for r in range(size) for c in range(size) if board[r][c] is None]
            if not empties:
                break
            r, c = random.choice(empties)
            state = self.place_move(game_id, r, c)
        return state

    def assign_ai(self, game_id: str, black_ai: str, white_ai: str) -> dict:
        """既存の対局（まだAIを割り当てていないもの）に、両方の色のAIを割り当てる。
        両方指定すると、割り当てた瞬間からサーバー側のAiVsAiDriverが自動で最後まで進める。"""
        resp = requests.post(
            f"{self.base_url}/api/games/{game_id}/ai",
            params={"blackAi": black_ai, "whiteAi": white_ai},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def get_history(self, game_id: str) -> list[dict]:
        resp = requests.get(f"{self.base_url}/api/games/{game_id}/history", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def wait_for_completion(self, game_id: str, poll_interval: float = 1.0, timeout: float = 180.0) -> list[dict]:
        """対局が終了する（historyの最後がgameOver=trueになる）まで待ってから履歴を返す。"""
        start = time.time()
        while True:
            history = self.get_history(game_id)
            if history and history[-1]["gameOver"]:
                return history
            if time.time() - start > timeout:
                raise TimeoutError(f"game {game_id} did not finish within {timeout}s")
            time.sleep(poll_interval)


class SimulationClient:
    """MCTS用のステートレスな1手シミュレーションAPI（/api/simulate/move）のクライアント。
    実際の対局（GameService登録）を一切介さず、盤面ルール（GameRoom.applyTurn）だけを呼ぶ。
    """

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")
        # MCTSは1局・1手あたり何度もこのAPIを叩くため、リクエストのたびに新しいTCP接続を
        # 張るrequests.post()の代わりにSessionでコネクションを使い回し、往復のオーバーヘッドを減らす。
        # ただし、並列ワーカーが多い/1手の探索に時間がかかる設定だと、プールに置かれた接続が
        # サーバー側のkeep-alive制限時間より長く放置され、次に使おうとした瞬間に向こうから
        # 既に切断されている（RemoteDisconnected）ことがある。/api/simulate/moveは対局登録を
        # 一切伴わないステートレスな処理なので、再試行しても安全 → 自動リトライで吸収する。
        self.session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.3,
            allowed_methods=frozenset(["GET", "POST"]),
            status_forcelist=(502, 503, 504),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def initial_state(self, size: int = 9) -> dict:
        """1手目より前の初期状態。GameRoom.reset()の初期値と一致させてある。"""
        return {
            "board": [[None] * size for _ in range(size)],
            "dmgMarks": [],
            "removalEchoes": {},
            "hp": {"B": 6, "W": 6},
            "pending": None,
            "currentPlayer": "B",
            "gameOver": False,
            "winner": None,
            "plyCount": 0,
            "markEventCount": 0,
            "markPerSide": 1,
            "nextMarkEventTurn": 10,
        }

    def simulate_move(self, state: dict, row: int, col: int) -> dict:
        resp = self.session.post(
            f"{self.base_url}/api/simulate/move",
            json={"state": state, "row": row, "col": col},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
