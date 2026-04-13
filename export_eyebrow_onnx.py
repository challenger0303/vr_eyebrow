import argparse
from pathlib import Path

import torch
import torch.nn as nn

from model import TinyBrowNet


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, mirror_legacy_output: bool) -> None:
        super().__init__()
        self.model = model
        self.mirror_legacy_output = mirror_legacy_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if output.dim() != 2:
            output = output.view(output.size(0), -1)

        if self.mirror_legacy_output:
            brow = output[:, :1]
            output = torch.cat((brow, brow, brow), dim=1)

        return output


def load_state_dict_compat(model: TinyBrowNet, checkpoint_path: Path) -> bool:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    current_dict = model.state_dict()
    mirror_legacy_output = False

    if "fc2.weight" in state_dict and "fc2.weight" in current_dict:
        out_old = state_dict["fc2.weight"].shape[0]
        out_new = current_dict["fc2.weight"].shape[0]
        if out_old != out_new:
            mirror_legacy_output = out_old == 1
            new_fc2_weight = current_dict["fc2.weight"].clone()
            new_fc2_bias = current_dict["fc2.bias"].clone()
            copy_n = min(out_old, out_new)
            new_fc2_weight[:copy_n, :] = state_dict["fc2.weight"][:copy_n, :]
            new_fc2_bias[:copy_n] = state_dict["fc2.bias"][:copy_n]
            state_dict["fc2.weight"] = new_fc2_weight
            state_dict["fc2.bias"] = new_fc2_bias

    model.load_state_dict(state_dict, strict=False)
    return mirror_legacy_output


def export_onnx(checkpoint_path: Path, output_path: Path, batch_size: int, opset: int) -> None:
    model = TinyBrowNet()
    mirror_legacy_output = load_state_dict_compat(model, checkpoint_path)
    wrapper = ExportWrapper(model.eval(), mirror_legacy_output).eval()
    output_width = model.fc2.out_features

    dummy_input = torch.randn(batch_size, 1, 64, 64, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=None,
            dynamo=False,
        )

    print(f"Exported ONNX model to: {output_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Legacy single-output mirrored: {mirror_legacy_output}")
    print(f"Native output width: {output_width}")
    print(f"Static batch size: {batch_size}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export TinyBrowNet checkpoints to a Baballonia-friendly ONNX model."
    )
    parser.add_argument(
        "--checkpoint",
        default="tinybrownet_best.pth",
        help="Path to the .pth checkpoint to export.",
    )
    parser.add_argument(
        "--output",
        default=(
            "../baballonia_eyebrow_integration/src/Baballonia/eyebrowModel.onnx"
        ),
        help="Path to the output ONNX file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Static batch size to bake into the ONNX graph. Baballonia expects 2 for L/R eyes.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    export_onnx(checkpoint_path, output_path, args.batch_size, args.opset)


if __name__ == "__main__":
    main()
