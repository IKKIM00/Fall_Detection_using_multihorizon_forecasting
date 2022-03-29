import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder

from model.LSTM import LSTM
from model.CNN import CNN

def choose_model(dataset, model_type, device):
    if model_type == 'singleLSTM':
        model = LSTM(6, 320, 100, 1).to(device)
        save_path = 'results/singleLSTM/'
        save_file_name = save_path + f'SingleLSTM{dataset}.pth'
    elif model_type == 'stackedLSTM':
        model = LSTM(6, 320, 100).to(device)
        save_path = 'results/stackedLSTM/'
        save_file_name = save_path + f'StackedLSTM{dataset}.pth'
    elif model_type == 'CNN':
        model = CNN(6, [80, 160, 320], 100).to(device)
        save_path = 'results/CNN/'
        save_file_name = save_path + f'CNN{dataset}.pth'
    return model, save_path, save_file_name


def fit(model, train_loader, valid_loader, optimizer, batch_size, n_epochs, criterion, model_type, save_file_name, device):
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    best_loss = 9999999999
    patience = 0

    for epoch in range(1, n_epochs + 1):
        ###################
        # train the model #
        ###################
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            if model_type != 'CNN':
                data, target = data.view(-1, 100, 6).to(device), target.to(device)
            else:
                data, target = data.to(device), target.to(device)
            output = model(data)
            if model_type != 'CNN':
                loss = criterion(output, target.squeeze())
            else:
                loss = criterion(output.squeeze(), target.squeeze())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        ######################
        # validate the model #
        ######################
        model.eval()
        for data, target in valid_loader:
            if model_type != 'CNN':
                data, target = data.view(-1, 100, 6).to(device), target.to(device)
            else:
                data, target = data.to(device), target.to(device)
            output = model(data.cuda())
            if model_type != 'CNN':
                loss = criterion(output, target.squeeze())
            else:
                loss = criterion(output.squeeze(), target.squeeze())
            valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        train_losses = []
        valid_losses = []

        if valid_loss < best_loss:
            best_loss = valid_loss
            patience = 0
            torch.save(model.state_dict(), save_file_name)
            print('Saving Model')
        else:
            patience += 1
            print('Patience for ', patience)
        if patience == 10:
            break

    model.load_state_dict(torch.load(save_file_name))

    return model, avg_train_losses, avg_valid_losses


def evaluate(model, test_loader, model_type, criterion):
    test_loss = 0.0
    y_test = []
    y_hat = []
    model.eval()
    for data, target in test_loader:
        if model_type != 'CNN':
            data, target = data.view(-1, 100, 6).cuda(), target.cuda()
        else:
            data, target = data.cuda(), target.cuda()
        output = model(data.cuda())
        loss = criterion(output.squeeze(), target.squeeze())
        test_loss += loss.item()
        y_test += list(target.squeeze().detach().cpu().numpy())
        y_hat += list(output.squeeze().detach().cpu().numpy())

    return test_loss / len(test_loader), y_test, y_hat



def preprocess_dlr(dataset_dir):
    train = pd.read_csv(dataset_dir + 'train.csv', index_col=0)
    valid = pd.read_csv(dataset_dir + 'valid.csv', index_col=0)
    test = pd.read_csv(dataset_dir + 'test.csv', index_col=0)

    activity_info = ['FALLING', 'JUMPING', 'RUNNING', 'SITTING', 'STNDING', 'TRANSDW', 'TRANSUP', 'TRNSACC', 'TRNSDCC',
                    'WALKING', 'XLYINGX']

    encoder = LabelEncoder()
    encoder.fit(activity_info)
    train_encoded = encoder.transform(train['labels'])
    train['label_encoded'] = train_encoded
    valid_encoded = encoder.transform(valid['labels'])
    valid['label_encoded'] = valid_encoded
    test_encoded = encoder.transform(test['labels'])
    test['label_encoded'] = test_encoded

    columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'label', 'per_idx']
    obs_scaler = StandardScaler()
    tar_scaler = StandardScaler()

    obs_train = train[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']]
    obs_train = obs_scaler.fit_transform(obs_train)
    tar_train = np.asarray(train['label_encoded'])
    tar_train = tar_scaler.fit_transform(tar_train.reshape(-1, 1))
    obs_train = pd.DataFrame(obs_train)
    tar_train = pd.DataFrame(tar_train)
    transformed_train = pd.concat([obs_train, tar_train], axis=1)
    transformed_train['per_idx'] = train['per_idx'].values
    transformed_train.columns = columns

    obs_valid = valid[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']]
    obs_valid = obs_scaler.transform(obs_valid)
    tar_valid = np.asarray(valid['label_encoded'])
    tar_valid = tar_scaler.transform(tar_valid.reshape(-1, 1))
    obs_valid = pd.DataFrame(obs_valid)
    tar_valid = pd.DataFrame(tar_valid)
    transformed_valid = pd.concat([obs_valid, tar_valid], axis=1)
    transformed_valid['per_idx'] = valid['per_idx'].values
    transformed_valid.columns = columns

    obs_test = test[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']]
    obs_test = obs_scaler.transform(obs_test)
    tar_test = np.asarray(test['label_encoded'])
    tar_test = tar_scaler.transform(tar_test.reshape(-1, 1))
    obs_test = pd.DataFrame(obs_test)
    tar_test = pd.DataFrame(tar_test)
    transformed_test = pd.concat([obs_test, tar_test], axis=1)
    transformed_test['per_idx'] = test['per_idx'].values
    transformed_test.columns = columns
    X_train, y_train = dlr_moving_window(transformed_train)
    X_valid, y_valid = dlr_moving_window(transformed_valid)
    X_test, y_test = dlr_moving_window(transformed_test)
    return X_train, y_train, X_valid, y_valid, X_test, y_test, obs_scaler, tar_scaler

def dlr_moving_window(data):
    X_data, y_data = list(), list()
    for i in range(100, len(data) - 100):
        acc_x = data['acc_x'][i - 100: i]
        acc_y = data['acc_y'][i - 100: i]
        acc_z = data['acc_z'][i - 100: i]
        gyro_x = data['gyro_x'][i - 100: i]
        gyro_y = data['gyro_y'][i - 100: i]
        gyro_z = data['gyro_z'][i - 100: i]
        X_data.append([acc_x.values, acc_y.values, acc_z.values, gyro_x.values, gyro_y.values, gyro_z.values])

        outcome = data['label'][i: i + 100]
        y_data.append([outcome.values])
    return X_data, y_data