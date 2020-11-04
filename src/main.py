import os
import json
import pickle
import argparse

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

from .vectorization import (load_dataset, vectorizes_features,
                            vectorizes_label, vectorizes_smile)

from .utils import get_app


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
    parser.add_argument("--model-type", choices=["dummy"], default="dummy")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--smile")
    parser.add_argument("--random-state", type=int,
                        help="integer value to seed the random generator",
                        default=1729)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    checked_args = check_argument_parsing(args)

    if checked_args.task == "train":
        df = load_dataset(args.fname)
        x, y = vectorizes_features(df), vectorizes_label(df)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=0.9,
            random_state=args.random_state
        )

        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        for arr, arr_name in ((x_test, "x_test.npy"), (y_test, "y_test.npy")):
            np.save(os.path.join(args.output_dir, arr_name), arr)

        if args.model_type == "dummy":
            clf = DummyClassifier(strategy="most_frequent")
            clf.fit(x_train, y_train)
            score = clf.score(x_test, y_test)

            with open(args.model, "wb") as f:
                pickle.dump(clf, f)

        res = {
            "model-type": args.model_type,
            "random_state": args.random_state,
            "score": score
        }
        with open(os.path.join(args.output_dir, "resultats.json"), "w") as f:
            json.dump(res, f)

    if args.model_type == "dummy":
        with open(args.model, "rb") as f:
            clf = pickle.load(f)
    if checked_args.task == "predict":
        try:
            x = vectorizes_smile(checked_args.smile)
            print("mol: {}, {}".format(args.smile, clf.predict(x)))
        except Exception() as e:
            print(e)

    elif checked_args.task == "evaluate":
        df = load_dataset(args.fname)
        x_test = vectorizes_features(df)
        y_test = vectorizes_label(df)
        score = clf.score(x_test, y_test)
        print(score)

    # API
    else:
        app = get_app(clf)
        app.run()


if __name__ == "__main__":
    main()
