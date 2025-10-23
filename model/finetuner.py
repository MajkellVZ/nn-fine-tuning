import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.data_generator import generate_sparse_new_data, make_loader
from model.mlp_regressor import MLP
from model.trainer import train_model, evaluate_model


def fine_tune_model(base_model_path: str, loader: DataLoader):
    model = MLP(in_dim=5, hidden=64)
    model.load_state_dict(torch.load(os.path.join("model_store", base_model_path)))
    model.freeze_feature_extractor()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    model = train_model(model, loader, criterion, optimizer)
    model.unfreeze_all()

    torch.save(model.state_dict(), os.path.join("model_store", "new_model.pt"))

    print("Fine tuned model saved successfully")
    return model

def run_fine_tuning():
    print('Fine tuning started')
    X_new, y_new = generate_sparse_new_data()
    X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(X_new, y_new, test_size=.2,
                                                                          random_state=42)

    train_loader_new = make_loader(X_train_new, y_train_new)

    model = fine_tune_model('base_model.pt', train_loader_new)
    mse = evaluate_model(model, X_val_new, y_val_new)

    print("Fine tuning completed")
    return mse