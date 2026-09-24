"""rl_selfplay.pyが集めたMCTS自己対戦データで学習する。yonmoku_nn.trainとの違いは、
方策の教師信号が「実際に打たれた1手」ではなく「MCTSの訪問回数分布」であること
（ソフトラベルなのでCrossEntropyLossではなく、log_softmaxとの内積で損失を計算する）。
"""

import argparse
import copy
import glob
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from .dataset import RLDataset
from .model import YonmokuNet


def soft_policy_loss(policy_logits: torch.Tensor, policy_target: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(policy_logits, dim=1)
    return -(policy_target * log_probs).sum(dim=1).mean()


def main():
    parser = argparse.ArgumentParser(description="MCTS自己対戦データからYonmokuNetを学習する（強化学習ループ用）")
    parser.add_argument("--data", nargs="+", default=["data/rl_selfplay*.jsonl"],
                         help="jsonlファイルのglobパターン（複数指定可）")
    parser.add_argument("--init-checkpoint", default=None,
                         help="学習の初期重み。指定が無ければランダム初期化から始める")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--out", default="checkpoints/model_rl.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    paths: list[str] = []
    for pattern in args.data:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        raise SystemExit(f"no data files matched: {args.data}")
    print(f"loading {len(paths)} file(s): {paths}")

    dataset = RLDataset(paths)
    if len(dataset) < 10:
        raise SystemExit(f"too few samples ({len(dataset)}) to train on")

    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    device = torch.device(args.device)
    model = YonmokuNet().to(device)
    if args.init_checkpoint and Path(args.init_checkpoint).exists():
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
        print(f"initialized from {args.init_checkpoint}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    value_loss_fn = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, policy_target, value in train_loader:
            x, policy_target, value = x.to(device), policy_target.to(device), value.to(device)
            optimizer.zero_grad()
            policy_logits, value_pred = model(x)
            loss = soft_policy_loss(policy_logits, policy_target) + value_loss_fn(value_pred, value)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, policy_target, value in val_loader:
                x, policy_target, value = x.to(device), policy_target.to(device), value.to(device)
                policy_logits, value_pred = model(x)
                loss = soft_policy_loss(policy_logits, policy_target) + value_loss_fn(value_pred, value)
                val_loss += loss.item() * x.size(0)
        avg_val_loss = val_loss / len(val_set)

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        marker = " *" if is_best else ""
        print(f"epoch {epoch + 1}/{args.epochs}: train_loss={avg_loss:.4f} val_loss={avg_val_loss:.4f}{marker}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_path)
    print(f"saved best model (epoch {best_epoch}, val_loss={best_val_loss:.4f}) to {out_path}")


if __name__ == "__main__":
    main()
