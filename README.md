# Machine-Learning Base Prediction of Molecule's Properties

This repository contains code for molecule proterty prediction,
based on neural-based inferences. There are two types of model:
* a multilayer perceptron with on intermediate layer
* a GRU-based RNN model

The RNN is learned with a model of [mol2vec](https://mol2vec.readthedocs.io/en/latest/)
 inspired by the famous [word2vec](https://fr.wikipedia.org/wiki/Word2vec) model.
Both model are written in tensorflow

## Installation

This package requires a working conda environment.
```
git clone adress
export PATH=~/miniconda/bin:$PATH
conda update -n base conda
conda create -y --name servier python=3.6
conda activate servier
conda install -c conda-forge rdkit
```

Then the entry point for each action is `servier`:
* `train` on a pandas dataframe with `--fname`
* `evaluate` on a pandas dataframe with `--fname`
* `predict` on a single smile
* `server-start` to launch a rest API (127.0.0.1:5000)

## Docker

It is possible to put this project in a docker
```
docker build . -t your_tag
```
The entry point is still `servier`.
```
docker run -it -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models -p 127.0.0.1:5000:5000 your_tag your_task
```

## Training

To train the model, a json file must be provided with the relevant hyperparameters
for each model. Currently the hyperparamets are:
* the number of units per layer `neurons`
* dropout rate `dropout_rate`
* the default activation for each layer of mlp `activation`
* the optimizer for gradient descent `opt`
* the number of epochs `epochs`
* the size for each batch `batch_size`

## Results Accuracy

DUMMY: 75%
MLP: 80%
RNN: 30%

## Next steps

The hyperparameter tuning was done by hand, one should continue do it more
rigorously with either a random search or with bayesian optimization,
the implementation could be done with the `hyperopt` packages.

The mol2vec model used was pretrained on the ZINC databases, I did
encounter some Out of Vocabulary problem, so retraining a mol2vec on
a database closer to our dataset could provide better results. There is a
problem of Out of Vocabulary, whcih probably has a big impact.

We are working on sequences representing molecules, but as there may exist some
cycle in the molecules, it may not be the method. Graph neural network, and 
espcially the use of the transformer's architecture could provide
good results.

As for the multiobjective tasks, I would like to read this
[paper](MONN: A Multi-objective Neural Network for Predicting Compound-Protein Interactions and Affinities) in more details.
