"""Download the datasets used by the experiments into ./data (or --root).

    python experiments/download_data.py                    # mnist, fashion_mnist, cifar10
    python experiments/download_data.py mnist --root /content/data

The experiments also download a dataset automatically on first use; this script just
does it up front (e.g. before going offline, or on Colab before a long run).
``digits`` and ``synthetic`` need no download. The data folder is in .gitignore.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flcourse.data import load_dataset  # noqa: E402

DATASETS = ["mnist", "fashion_mnist", "cifar10"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", default=DATASETS, choices=DATASETS + ["digits"])
    parser.add_argument("--root", default="./data", help="download folder (= data.root in the configs)")
    args = parser.parse_args(argv)
    for name in args.datasets:
        print(f"{name}: downloading to {args.root} ...", flush=True)
        train, test = load_dataset({"dataset": name, "root": args.root})
        print(f"{name}: ok  train={len(train)}  test={len(test)}  input shape={train.input_shape}")


if __name__ == "__main__":
    main()
