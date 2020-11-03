import numpy as np
import pandas as pd

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


def load_dataset(fname: str) -> pd.DataFrame:
    return pd.read_csv(fname)


def vectorizes_label(df: pd.DataFrame) -> np.ndarray:
    return df["P1"].values.reshape((-1, 1))


def vectorizes_features(df: pd.DataFrame, **fingerprint_args) -> np.ndarray:
    smiles = df["smiles"].apply(
        lambda x: np.array(fingerprint_features(x, **fingerprint_args))
    )
    return np.vstack(smiles.values)
