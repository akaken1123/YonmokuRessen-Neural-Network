import json

import numpy as np
import torch
from torch.utils.data import Dataset

from .encoding import SIZE, encode_state


class SelfPlayDataset(Dataset):
    """selfplay.pyが書き出したJSONLファイル群を読み込むDataset。"""

    def __init__(self, jsonl_paths: list[str]):
        self.samples: list[dict] = []
        for path in jsonl_paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        perspective = sample["move"]["color"]
        x = encode_state(sample["state_before"], perspective)

        move_idx = sample["move"]["row"] * SIZE + sample["move"]["col"]

        winner = sample["winner"]
        if winner == perspective:
            value = 1.0
        elif winner is None or winner == "draw":
            value = 0.0
        else:
            value = -1.0

        return (
            torch.from_numpy(x),
            torch.tensor(move_idx, dtype=torch.long),
            torch.tensor(value, dtype=torch.float32),
        )


class RLDataset(Dataset):
    """rl_selfplay.pyが書き出したJSONLファイル群を読み込むDataset。
    方策ターゲットは実際に打たれた1手（argmax）ではなく、MCTSの訪問回数分布を正規化した
    確率ベクトル（81次元、ソフトラベル）にする。"""

    def __init__(self, jsonl_paths: list[str]):
        self.samples: list[dict] = []
        for path in jsonl_paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        perspective = sample["mover"]
        x = encode_state(sample["state_before"], perspective)

        policy_target = np.zeros(SIZE * SIZE, dtype=np.float32)
        visit_counts = sample["visit_counts"]
        total = sum(visit_counts.values())
        if total > 0:
            for move_idx_str, count in visit_counts.items():
                policy_target[int(move_idx_str)] = count / total

        winner = sample["winner"]
        if winner == perspective:
            value = 1.0
        elif winner is None or winner == "draw":
            value = 0.0
        else:
            value = -1.0

        return (
            torch.from_numpy(x),
            torch.from_numpy(policy_target),
            torch.tensor(value, dtype=torch.float32),
        )


class CombinedDataset(Dataset):
    """selfplay.py（模倣学習、実際に打たれた1手）とrl_selfplay.py（MCTSの訪問回数分布）の
    両方のJSONLファイルを同じDatasetで扱えるようにしたもの。1行ごとに"move"キーがあれば
    模倣学習のサンプル（方策ターゲットはその1手だけのone-hotベクトル）、"visit_counts"キーが
    あればRLのサンプル（方策ターゲットは訪問回数分布）として扱う。one-hotはソフトラベルの
    特別な場合（1箇所だけ確率1.0）なので、どちらもsoft_policy_loss（train_rl.py）でそのまま
    学習できる。

    これは、RL自己対戦のたびに模倣学習で作った重みを少しずつ上書きしていく方式（train_rl.pyの
    従来の使い方）が、模倣学習で積み上げた実力を毎回壊してしまう問題への対策。模倣データと
    蓄積した自己対戦データを毎回まとめて、模倣学習と同じくらい本格的に（多いエポック数で
    val_lossに基づくベスト選択をしながら）学習し直すことで、少量の自己対戦データだけで
    大きく重みを動かしてしまうことを避ける。
    """

    def __init__(self, jsonl_paths: list[str]):
        self.samples: list[dict] = []
        for path in jsonl_paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        is_imitation = "move" in sample
        perspective = sample["move"]["color"] if is_imitation else sample["mover"]
        x = encode_state(sample["state_before"], perspective)

        policy_target = np.zeros(SIZE * SIZE, dtype=np.float32)
        if is_imitation:
            move_idx = sample["move"]["row"] * SIZE + sample["move"]["col"]
            policy_target[move_idx] = 1.0
        else:
            visit_counts = sample["visit_counts"]
            total = sum(visit_counts.values())
            if total > 0:
                for move_idx_str, count in visit_counts.items():
                    policy_target[int(move_idx_str)] = count / total

        winner = sample["winner"]
        if winner == perspective:
            value = 1.0
        elif winner is None or winner == "draw":
            value = 0.0
        else:
            value = -1.0

        return (
            torch.from_numpy(x),
            torch.from_numpy(policy_target),
            torch.tensor(value, dtype=torch.float32),
        )
