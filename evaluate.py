from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from eval import evaluate


if __name__ == "__main__":
    parser = ArgumentParser(description="Fiery evaluation")
    parser.add_argument("--checkpoint", default="./fiery.ckpt", type=str, help="path to checkpoint")
    parser.add_argument("--dataroot", default="./data/nuscenes", type=str, help="path to the dataset")
    parser.add_argument(
        "--version",
        default="trainval",
        type=str,
        choices=["mini", "trainval"],
        help="dataset version",
    )
    parser.add_argument("--device", default="cuda:0", type=str, help="evaluation device")

    args = parser.parse_args()

    evaluate(args.checkpoint, args.dataroot, args.version, args.device)
