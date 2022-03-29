import argparse

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from libs.dl_utils import *
from model.LSTM import LSTM
from model.CNN import CNN

np.random.seed(42)

def main(expt_name, model_type, batch_size=256, lr=0.01, n_epochs=300, use_gpu='yes'):

    if use_gpu == 'yes' and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    dataset_dirs = {
        'mobiact': 'dataset/mobiact_preprocessed/',
        'dlr': 'dataset/dlr_preprocessed/',
        'notch': 'dataset/Notch_Dataset/',
        'smartfall': 'dataset/SmartFall_Dataset/'
    }

    get_dataset = {
        'mobiact': get_mobiact(dataset_dirs[expt_name]),
        'dlr': get_dlr(dataset_dirs[expt_name]),

    }

    X_train, y_train, X_valid, y_valid, X_test, y_test, obs_scaler, tar_scaler = get_dataset[expt_name]

    train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    valid_data = TensorDataset(torch.Tensor(X_valid), torch.Tensor(y_valid))
    test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, num_workers=2)
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, num_workers=2)

    singlelstm_params = {
        'mobiact': [6, 320, 32],
        'dlr': [6, 320, 100],
        'notch': [3, 320, 32],
        'smartfall': [3, 320, 32]
    }
    stackedlstm_params = {
        'mobiact': [6, 320, 32],
        'dlr': [6, 320, 100],
        'notch': [3, 320, 32],
        'smartfall': [3, 320, 32]
    }
    cnn_params = {
        'mobiact': [6, [8, 16, 64], 32],
        'dlr': [6, [80, 160, 320], 100],
        'notch': [3, [8, 16, 64], 32],
        'smartfall': [3, [80, 160, 320], 32]
    }
    def choose_model(model_type):
        if model_type == 'singleLSTM':
            params = singlelstm_params[expt_name]
            model = LSTM(params[0], params[1], params[2], num_layers=1).to(device)
            save_path = 'results/singleLSTM/'
            save_file_name = f'{save_path}singleLSTM_{expt_name}.pth'
        elif model_type == 'stackedLSTM':
            params = stackedlstm_params[expt_name]
            model = LSTM(params[0], params[1], params[2], num_layers=2).to(device)
            save_path = 'results/stackedLSTM/'
            save_file_name = f'{save_path}stackedLSTM_{expt_name}.pth'
        elif model_type == 'CNN':
            params = cnn_params[expt_name]
            model = CNN(params[0], params[1], params[2]).to(device)
            save_path = 'results/CNN/'
            save_file_name = f'{save_path}CNN_{expt_name}.pth'
        return model, save_path, save_file_name

    model, save_path, save_file_name = choose_model(model_type)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           mode='min',
                                                           factor=0.2
                                                           )
    criterion = nn.MSELoss().to(device)

    model, train_loss, valid_loss = fit(model, model_type, train_loader, valid_loader, optimizer, scheduler, criterion, n_epochs, device, save_file_name)
    test_loss, y_target, y_hat = evaluate_model(model, model_type, criterion, test_loader, device)

    print('Fianl test loss: ', test_loss)
    np.savez(f'{save_path}predicted_{expt_name}.npz')
    print(f'Saved predicted value in {save_path}')


if __name__=='__main__':
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            'expt_name',
            type=str,
            default='mobiact'
        )
        parser.add_argument(
            'model_type',
            type=str,
            default='singeLSTM'
        )
        parser.add_argument(
            'use_gpu',
            type=str,
            choices=['yes', 'no'],
            default='yes'
        )
        args = parser.parse_known_args()[0]
        return args.expt_name, args.model_type, args.use_gpu == "yes"
    expt_name, model_type, use_gpu = get_args()

    main(expt_name, model_type, use_gpu=use_gpu)
