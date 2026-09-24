import json

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
