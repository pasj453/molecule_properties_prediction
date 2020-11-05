from typing import List
import tensorflow as tf
import numpy as np
import pandas as pd

from rdkit import Chem
from mol2vec.features import mol2alt_sentence
from rdkit.Chem import rdMolDescriptors, MolFromSmiles, rdmolfiles, rdmolops


def fingerprint_features(smile_string, radius=2, size=2048):
    mol = MolFromSmiles(smile_string)
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius,
                                                          nBits=size,
                                                          useChirality=True,
                                                          useBondTypes=True,
                                                          useFeatures=False
                                                          )


def vectorizes_smile(smile: str) -> np.ndarray:
    return np.array(fingerprint_features(smile)).reshape((1, -1))


def get_embbed(x, mol2vec):
    try:
        return mol2vec.wv.word_vec(x)
    except KeyError:
        return np.zeros(300)


def vec_mol2vec_smile(smiles: List[str], mol2vec) -> np.ndarray:
    # TODO evaluate impact of radius
    alt_seqs = map(lambda x: mol2alt_sentence(Chem.MolFromSmiles(x), 1),
                   smiles)
    vec_seqs = []
    for seqs in alt_seqs:
        vec_seqs.append(
            [get_embbed(x, mol2vec) for x in seqs]
        )
    return tf.keras.preprocessing.sequence.pad_sequences(
        vec_seqs, padding="post", truncating="post", dtype="float32"
    )


def load_dataset(fname: str) -> pd.DataFrame:
    return pd.read_csv(fname)


def vectorizes_label(df: pd.DataFrame) -> np.ndarray:
    return df["P1"].values.reshape((-1, 1))


def vectorizes_features(df: pd.DataFrame, **fingerprint_args) -> np.ndarray:
    smiles = df["smiles"].apply(
        lambda x: np.array(fingerprint_features(x, **fingerprint_args))
    )
    return np.vstack(smiles.values)
