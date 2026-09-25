"""学習済みモデルをONNX形式でエクスポートする。
Java側での推論統合（onnxruntime-java等の利用）は今後の作業。
"""

import argparse

import torch

from .encoding import NUM_PLANES, SIZE
from .model import YonmokuNet


def main():
    parser = argparse.ArgumentParser(description="学習済みモデルをONNX形式でエクスポートする")
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--out", default="checkpoints/model.onnx")
    args = parser.parse_args()

    model = YonmokuNet()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    dummy_input = torch.zeros(1, NUM_PLANES, SIZE, SIZE)
    torch.onnx.export(
        model,
        dummy_input,
        args.out,
        input_names=["board"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "board": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
        # 新しいdynamoベースのエクスポーター（torchのバージョンによってはデフォルトが
        # dynamo=Trueになっている）は、この程度の小さいモデルでも重みを別ファイル
        # （model.onnx.data）に分けて書き出そうとし、その外部データ書き出し処理に
        # Windows環境で失敗する既知の不具合がある。dynamic_axesもdynamo=False（従来の
        # TorchScriptベースのエクスポーター）向けの引数なので、明示的にdynamo=Falseを
        # 指定し、単一の.onnxファイルにまとめて書き出す。
        dynamo=False,
    )
    print(f"exported ONNX model to {args.out}")


if __name__ == "__main__":
    main()
