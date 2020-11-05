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

## Results
