from gensim.models import word2vec
from flask import Flask
from .vectorization import vectorizes_smile, vec_mol2vec_smile


def get_app(model, model_type, mol2vec=None):
    app = Flask("Servier Molecule Prediction")

    mol2vec = word2vec.Word2Vec.load('models/model_300dim.pkl')

    @app.route("/predict/<smile>")
    def predict(smile):
        try:
            if model_type in ["dummy", "mlp"]:
                x = vectorizes_smile(smile)
            else:
                x = vec_mol2vec_smile([smile], mol2vec)
            return "Properties prediction: {}".format(model.predict(x)[0])
        except Exception as e:
            print(e)

    return app
