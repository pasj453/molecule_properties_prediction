from flask import Flask
from .vectorization import vectorizes_smile


def get_app(model):
    app = Flask("Servier Molecule Prediction")

    @app.route("/predict/<smile>")
    def predict(smile):
        try:
            x = vectorizes_smile(smile)
            return "Properties prediction: {}".format(model.predict(x)[0])
        except Exception as e:
            print(e)

    return app
