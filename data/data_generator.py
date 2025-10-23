from typing import Tuple
import numpy as np
import torch
from sklearn.datasets import make_regression
from torch.utils.data import DataLoader, TensorDataset


def generate_initial_data(n_samples: int = 200, n_features: int = 5)-> Tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        noise=10,
        random_state=42
    )
    return X, y

def generate_sparse_new_data(n_samples: int = 50, n_features: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        noise=14,
        random_state=123
    )

    X = X * 1.1 + 0.5
    y = y * 1.05 + 10

    return X, y

def make_loader(X: np.ndarray, y: np.ndarray, batch: int = 32, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)),
                       torch.from_numpy(y.astype(np.float32)).unsqueeze(1))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)
