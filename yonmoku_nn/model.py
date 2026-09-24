import torch
import torch.nn as nn

from .encoding import NUM_PLANES, SIZE


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


class YonmokuNet(nn.Module):
    """9x9盤向けの小さな方策+価値ネットワーク（AlphaZero系アーキテクチャの簡易版）。"""

    def __init__(self, in_planes: int = NUM_PLANES, channels: int = 64, blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * SIZE * SIZE, SIZE * SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(SIZE * SIZE, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy_logits, value
