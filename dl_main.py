import pandas as pd
import numpy as np
import argparse

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from libs.dl_utils import choose_model, fit, evaluate, preprocess_dlr

def main(dataset, model_type, restart, batch_size):

    np.random.seed(42)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    dataset_dir = {
        'dlr': 'dataset/dlr_preprocessed/'
    }
    preprocess = {
        'dlr': preprocess_dlr(dataset_dir['dlr'])
    }

    X_train, y_train, X_valid, y_valid, X_test, y_test, obs_scaler, tar_scaler = preprocess[dataset]
    train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)

    valid_data = TensorDataset(torch.Tensor(X_valid), torch.Tensor(y_valid))
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)

    test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    test_loader = DataLoader(test_data, shuffle=False, batch_size=1)

    model, save_path, save_file_name = choose_model(dataset, model_type, device)
    print(model)

    lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           mode='min',
                                                           factor=0.2
                                                           )
    criterion = nn.MSELoss().to(device)
    patience = 5
    n_epochs = 300

    model, train_loss, valid_loss = fit(model, train_loader, valid_loader, optimizer, batch_size, n_epochs, criterion, model_type, save_file_name, device)
    test_loss, y_target, y_hat = evaluate(model, test_loader, model_type, criterion)

    labels = tar_scaler.inverse_transform(y_target)
    predicted = tar_scaler.inverse_transform(y_hat)


