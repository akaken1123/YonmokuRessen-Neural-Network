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
    )
    print(f"exported ONNX model to {args.out}")


if __name__ == "__main__":
    main()
