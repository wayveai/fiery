from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from visualise import visualise


if __name__ == "__main__":
    parser = ArgumentParser(description="Fiery visualisation")
    parser.add_argument("--checkpoint", default="./fiery.ckpt", type=str, help="path to checkpoint")
    parser.add_argument("--output-dir", default="./output_vis", type=str, help="output directory")
    parser.add_argument("--example-data-dir", default="./data/example_data", type=str, help="example data directory")

    args = parser.parse_args()

    visualise(
        checkpoint_path=args.checkpoint,
        output_dir=Path(args.output_dir),
        example_data_dir=Path(args.example_data_dir),
    )
