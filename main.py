import os
import argparse


def check_argument_parsing(args: argparse.Namespace) -> argparse.Namespace:
    if args.task == "train" or args.task == "evaluate":
        if not args.fname or not os.path.isfile(args.fname):
            raise FileNotFoundError(
                "Dataset is missing for {}".format(args.task)
            )

    if args.task != "train":
        if not os.path.isfile(args.model):
            raise FileNotFoundError(
                "For prediction a working model must be supplied"
            )

    if args.task == "predict":
        if not args.smile:
            raise Exception("Smile string must be returned for prediction")
    return args


def get_parser() -> argparse.ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task", choices=["train", "evaluate", "predict", "server-start"]
    )
    parser.add_argument(
        "--objective", choices=["single-metrics", "multi-metrics"],
        help="Performs single metric or multi-metric optimization",
        default="single-metrics",
    )
    parser.add_argument("--fname", help="filepath for dataset")
    parser.add_argument("--model", help="filepath for model loading or saving",
                        default="models/model.h5")
    parser.add_argument("--smile")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    checked_args = check_argument_parsing(args)
