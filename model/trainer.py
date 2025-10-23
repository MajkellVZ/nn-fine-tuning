import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.data_generator import generate_initial_data, make_loader
from model.mlp_regressor import MLP


def train_model(model: nn.Module, loader: DataLoader, criterion, optimizer, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    for ep in range(epochs):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
    return model

def evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(X.astype(np.float32))
        pred = model(x_tensor).squeeze().numpy()
        mse = mean_squared_error(y, pred)
    return mse

def run_model_training():
    X_initial, y_initial = generate_initial_data()
    X_train_init, X_val_init, y_train_init, y_val_init = train_test_split(X_initial, y_initial, test_size=.2,
                                                                          random_state=42)
    train_loader_init = make_loader(X_train_init, y_train_init)

    model = MLP(5, 64)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = train_model(model, train_loader_init, criterion, optimizer)
    mse = evaluate_model(model, X_val_init, y_val_init)

    os.makedirs("model_store", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("model_store", "base_model.pt"))

    print("Model training completed")
    return mse
