import argparse
import copy
import glob
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from .dataset import SelfPlayDataset
from .model import YonmokuNet


def main():
    parser = argparse.ArgumentParser(description="自己対戦データからYonmokuNetを学習する")
    parser.add_argument("--data", nargs="+", default=["data/*.jsonl"],
                         help="jsonlファイルのglobパターン（複数指定可）")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                         help="Adamのweight decay。過学習を抑えるための正則化")
    parser.add_argument("--out", default="checkpoints/model.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    paths: list[str] = []
    for pattern in args.data:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        raise SystemExit(f"no data files matched: {args.data}")
    print(f"loading {len(paths)} file(s): {paths}")

    dataset = SelfPlayDataset(paths)
    if len(dataset) < 10:
        raise SystemExit(f"too few samples ({len(dataset)}) to train on")

    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    device = torch.device(args.device)
    model = YonmokuNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    policy_loss_fn = torch.nn.CrossEntropyLoss()
    value_loss_fn = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, move_idx, value in train_loader:
            x, move_idx, value = x.to(device), move_idx.to(device), value.to(device)
            optimizer.zero_grad()
            policy_logits, value_pred = model(x)
            loss = policy_loss_fn(policy_logits, move_idx) + value_loss_fn(value_pred, value)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, move_idx, value in val_loader:
                x, move_idx, value = x.to(device), move_idx.to(device), value.to(device)
                policy_logits, value_pred = model(x)
                loss = policy_loss_fn(policy_logits, move_idx) + value_loss_fn(value_pred, value)
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
    # 最終エポックではなく、検証lossが最も良かった時点の重みを保存する（過学習したものを使わないため）。
    torch.save(best_state, out_path)
    print(f"saved best model (epoch {best_epoch}, val_loss={best_val_loss:.4f}) to {out_path}")


if __name__ == "__main__":
    main()
