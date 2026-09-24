"""起動中のYonmokuRessen Javaサーバーと通信する薄いREST APIクライアント。
盤面ルールはサーバー側（Java）の実装が唯一の正解であり、ここでは一切再実装しない。
"""

import time

import requests


class YonmokuClient:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")

    def create_game(self, black_ai: str, white_ai: str) -> str:
        """blackAi/whiteAiを両方指定すると、サーバー側のAiVsAiDriverが自動で最後まで対局を進める。"""
        resp = requests.post(
            f"{self.base_url}/api/games",
            params={"blackAi": black_ai, "whiteAi": white_ai},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["id"]

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
        resp = requests.post(
            f"{self.base_url}/api/simulate/move",
            json={"state": state, "row": row, "col": col},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
