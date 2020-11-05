import os
import json
import pickle
import argparse

import numpy as np
import tensorflow as tf
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

from .vectorization import (load_dataset, vectorizes_features,
                            vectorizes_label, vectorizes_smile,
                            vec_mol2vec_smile)

from .utils import get_app

from .models import get_mlp, get_callbacks, get_rnn


def check_argument_parsing(args: argparse.Namespace) -> argparse.Namespace:
    """ parse command line arguments """
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
    parser.add_argument("--model2vec", help="filepath for mol2vec",
                        default="models/models/model_300dim.pkl")
    parser.add_argument("--model-type", choices=["dummy", "mlp", "rnn"],
                        default="dummy")
    parser.add_argument("--hyperparameters", default="hp.json",
                        help="filepath for model hyperparameters")
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

    multi = True if args.objective == "multi-metrics" else False

    if checked_args.task == "train":
        df = load_dataset(args.fname)

        # vectorization
        if not (args.model_type == "rnn"):
            x, y = vectorizes_features(df), vectorizes_label(df, multi)
        else:
            mol2vec = word2vec.Word2Vec.load('models/model_300dim.pkl')
            y = vectorizes_label(df, multi)
            x = vec_mol2vec_smile(df["smiles"].tolist(), mol2vec)

        # train-test-split
        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            stratify=y if not multi else None,
            train_size=0.9,
            random_state=args.random_state
        )

        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        # useful for restesting on same array
        for arr, arr_name in ((x_test, "x_test.npy"), (y_test, "y_test.npy")):
            np.save(os.path.join(args.output_dir, arr_name), arr)

        if args.model_type == "dummy":
            clf = DummyClassifier(strategy="most_frequent")
            clf.fit(x_train, y_train)
            score = clf.score(x_test, y_test)

            with open(args.model, "wb") as f:
                pickle.dump(clf, f)

        elif args.model_type == "mlp":
            with open(args.hyperparameters) as f:
                hp = json.load(f)
            clf = get_mlp(1 if args.objective == "single-metrics" else 9,
                          hp["neurons"], hp["dropout_rate"],
                          hp["activation"])
            clf.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=hp["learning_rate"]
                ),
                loss="categorical_crossentropy",
                metrics=['accuracy']
            )
            clf.fit(x_train, y_train, batch_size=hp["batch_size"],
                    epochs=hp["epochs"], validation_data=(x_test, y_test),
                    callbacks=get_callbacks(args.model))
            score = clf.evaluate(x_test, y_test,
                                 batch_size=hp["batch_size"])[1]

        elif args.model_type == "rnn":
            with open(args.hyperparameters) as f:
                hp = json.load(f)
            clf = get_rnn(1 if args.objective == "single-metrics" else 9,
                          hp["neurons"], hp["dropout_rate"])
            clf.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=hp["learning_rate"]
                ),
                loss="categorical_crossentropy",
                metrics=['accuracy']
            )
            clf.fit(x_train, y_train, batch_size=hp["batch_size"],
                    epochs=hp["epochs"], validation_data=(x_test, y_test),
                    callbacks=get_callbacks(args.model))
            score = clf.evaluate(x_test, y_test,
                                 batch_size=hp["batch_size"])[1]

        results = {
            "model-type": args.model_type,
            "random_state": args.random_state,
            "score": float(score)
        }
        with open(os.path.join(args.output_dir, "resultats.json"), "w") as f:
            json.dump(results, f)

    else:
        if args.model_type == "dummy":
            with open(args.model, "rb") as f:
                clf = pickle.load(f)
        elif args.model_type == "mlp":
            clf = tf.keras.Model.load_model(args.model)
        elif args.model_type == "rnn":
            clf = tf.keras.load_model(args.model)
            mol2vec = word2vec.Word2Vec.load('models/model_300dim.pkl')

        if checked_args.task == "predict":
            if args.model_type in ["dummy", "mlp"]:
                try:
                    x = vectorizes_smile(checked_args.smile)
                    print("mol: {}, {}".format(args.smile, clf.predict(x)))
                except Exception() as e:
                    print(e)
            else:
                x = vec_mol2vec_smile([args.smile], mol2vec)

        if checked_args.task == "evaluate":
            df = load_dataset(args.fname)
            if args.model_type == "rnn":
                x_test = vec_mol2vec_smile(df["smiles"].tolist())
            else:
                x_test = vectorizes_features(df)
            y_test = vectorizes_label(df, multi)
            score = clf.score(x_test, y_test)
            print(score)

        # API
        elif checked_args.task == "server-start":
            app = get_app(clf, args.model_type, args.mol2vec)
            app.run()


if __name__ == "__main__":
    main()
