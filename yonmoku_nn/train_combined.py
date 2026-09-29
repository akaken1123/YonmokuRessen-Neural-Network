"""模倣学習データ（selfplay.py、実際に打たれた1手）とRL自己対戦データ（rl_selfplay.py、
MCTSの訪問回数分布）を同じ学習データとしてまとめて、model_base系と同じくらい本格的に
（多いエポック数・val_lossに基づくベスト選択で）学習し直す。

train_rl.pyの「毎世代、少量の自己対戦データだけで軽く追加学習する」方式は、模倣学習で
積み上げた実力（model_base5.pt、DEFAULTへの勝率35〜50%程度）を毎回壊してしまうことが
実測で分かった（学習率を1e-4→1e-5、エポック数を8→2まで下げても改善せず、RL世代は毎回
例外なく劣化）。原因は、13,000局規模の模倣データで慎重に学習した重みを、150局程度の
小さく偏った自己対戦データで、たった数エポックだけ上書きしてしまっていたことだと考えられる。

このスクリプトでは、蓄積した自己対戦データ（世代を重ねるごとに--dataのglobに追加していく）を
模倣データと一緒に、model.pyのtrain.py/train_rl.pyと同じ学習レシピ（既定エポック数40、
val_lossが最良のエポックを保存）で学習し直す。これにより、小さい自己対戦データだけで
重みが大きく歪められることを避ける。
"""

import argparse
import copy
import glob
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from .dataset import CombinedDataset
from .model import YonmokuNet
from .train_rl import soft_policy_loss


def main():
    parser = argparse.ArgumentParser(
        description="模倣学習データ+RL自己対戦データをまとめて本格的に学習し直す（RLの破滅的忘却対策）")
    parser.add_argument("--data", nargs="+",
                         default=["data/selfplay_test3_v*.jsonl", "data/rl_buffer/cycle*.jsonl"],
                         help="jsonlファイルのglobパターン（複数指定可、模倣データとRLデータを両方含める）")
    parser.add_argument("--init-checkpoint", default=None,
                         help="学習の初期重み。指定が無ければランダム初期化から始める")
    parser.add_argument("--epochs", type=int, default=40,
                         help="model_base系と同じ既定値。少量データの微調整ではないので、"
                              "きちんと収束するまで多めに回す")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3,
                         help="model_base系と同じ既定値。train_rl.pyの微調整用（1e-4〜1e-5）とは違い、"
                              "ゼロから本格的に学習し直すのでtrain.pyと同じ学習率を使う")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--out", default="checkpoints/model_combined.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    paths: list[str] = []
    for pattern in args.data:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        raise SystemExit(f"no data files matched: {args.data}")
    print(f"loading {len(paths)} file(s): {paths}")

    dataset = CombinedDataset(paths)
    if len(dataset) < 10:
        raise SystemExit(f"too few samples ({len(dataset)}) to train on")
    print(f"loaded {len(dataset)} sample(s)")

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
