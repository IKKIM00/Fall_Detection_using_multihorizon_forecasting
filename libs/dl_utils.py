import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

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


def fit(model, model_type, train_loader, valid_loader, optimizer, scheduler, criterion, n_epochs, device, save_file_name):
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
            # print(data.shape, target.shape)
            optimizer.zero_grad()
            if model_type in ['SingleLSTM', 'StackedLSTM']:
                data = data.permute(0, 2, 1).contiguous()
            output = model(data.to(device))
            if model_type == 'CNN':
                output = output.squeeze()
            loss = criterion(output, target.to(device).squeeze())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        ######################
        # validate the model #
        ######################
        model.eval()
        for data, target in valid_loader:
            if model_type in ['SingleLSTM', 'StackedLSTM']:
                data = data.permute(0, 2, 1).contiguous()
            output = model(data.to(device))
            if model_type == 'CNN':
                output = output.squeeze()
            loss = criterion(output, target.to(device).squeeze())
            valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        scheduler.step(valid_loss)

        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        print(print_msg)

        train_losses, valid_losses = [], []

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


def evaluate_model(model, model_type, criterion, test_loader, device):
    test_loss = 0.0
    y_test, y_hat = [], []

    model.eval()
    for data, target in test_loader:
        if model_type in ['SingleLSTM', 'StackedLSTM']:
            data = data.permute(0, 2, 1).contiguous()
        output = model(data.to(device))
        if model_type == 'CNN':
            output = output.squeeze()
        loss = criterion(output, target.to(device).squeeze())
        test_loss += loss.item()
        y_test += list(target.squeeze().detach().cpu().numpy())
        y_hat += list(output.squeeze().detach().cpu().numpy())

    return test_loss / len(test_loader), np.array(y_test), np.array(y_hat)


def dataProcessing(data, tw, label_name, c_num=6):
    X_data, y_data = list(), list()
    for i in range(tw, len(data) - tw):
        acc_x = data['acc_x'][i - tw: i]
        acc_y = data['acc_y'][i - tw: i]
        acc_z = data['acc_z'][i - tw: i]
        if c_num == 6:
            gyro_x = data['gyro_x'][i - tw: i]
            gyro_y = data['gyro_y'][i - tw: i]
            gyro_z = data['gyro_z'][i - tw: i]
            X_data.append([acc_x.values, acc_y.values, acc_z.values, gyro_x.values, gyro_y.values, gyro_z.values])
        else:
            X_data.append([acc_x.values, acc_y.values, acc_z.values])

        outcome = data[label_name][i: i + tw]
        y_data.append([outcome.values])
    return X_data, y_data

def get_mobiact(dataset_dir):

    train = pd.read_csv(dataset_dir + 'train.csv')
    valid = pd.read_csv(dataset_dir + 'valid.csv')
    test = pd.read_csv(dataset_dir + 'test.csv')
    activity_info = pd.read_csv(dataset_dir + 'activity_info.csv', index_col=0)

    encoder = LabelEncoder()
    encoder.fit(activity_info['Label'])

    train_encoded = encoder.transform(train['label'])
    train['label_encoded'] = train_encoded

    valid_encoded = encoder.transform(valid['label'])
    valid['label_encoded'] = valid_encoded

    test_encoded = encoder.transform(test['label'])
    test['label_encoded'] = test_encoded

    obs_scaler = StandardScaler()
    tar_scaler = StandardScaler()

    columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'label', 'label_encoded']
    obs_train = train[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']]
    obs_train = obs_scaler.fit_transform(obs_train)
    tar_train = np.asarray(train['label_encoded'])
    tar_train = tar_scaler.fit_transform(tar_train.reshape(-1, 1))

    obs_train = pd.DataFrame(obs_train)
    tar_train = pd.DataFrame(tar_train)
    transformed_train = pd.concat([obs_train, tar_train], axis=1)
    transformed_train.columns = columns
    transformed_train['per-id'] = train['person_id'].values

    obs_valid = valid[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']]
    obs_valid = obs_scaler.transform(obs_valid)
    tar_valid = np.asarray(valid['label_encoded'])
    tar_valid = tar_scaler.transform(tar_valid.reshape(-1, 1))

    obs_valid = pd.DataFrame(obs_valid)
    tar_valid = pd.DataFrame(tar_valid)
    transformed_valid = pd.concat([obs_valid, tar_valid], axis=1)
    transformed_valid.columns = columns
    transformed_valid['per-id'] = valid['person_id'].values

    obs_test = test[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']]
    obs_test = obs_scaler.transform(obs_test)
    tar_test = np.asarray(test['label_encoded'])
    tar_test = tar_scaler.transform(tar_test.reshape(-1, 1))

    obs_test = pd.DataFrame(obs_test)
    tar_test = pd.DataFrame(tar_test)
    transformed_test = pd.concat([obs_test, tar_test], axis=1)
    transformed_test.columns = columns
    transformed_test['per-id'] = test['person_id'].values

    X_train, y_train = dataProcessing(transformed_train, label_name='label_encoded', tw=43)
    X_valid, y_valid = dataProcessing(transformed_valid, label_name='label_encoded', tw=43)
    X_test, y_test = dataProcessing(transformed_test, label_name='label_encoded', tw=43)
    return X_train, y_train, X_valid, y_valid, X_test, y_test, obs_scaler, tar_scaler

def get_dlr(dataset_dir):
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

    obs_scaler = StandardScaler()
    tar_scaler = StandardScaler()

    columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'label', 'label_encoded']

    obs_train = train[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']]
    obs_train = obs_scaler.fit_transform(obs_train)
    tar_train = np.asarray(train['label_encoded'])
    tar_train = tar_scaler.fit_transform(tar_train.reshape(-1, 1))

    obs_train = pd.DataFrame(obs_train)
    tar_train = pd.DataFrame(tar_train)
    transformed_train = pd.concat([obs_train, tar_train], axis=1)
    transformed_train.columns = columns
    transformed_train['per_idx'] = train['per_idx'].values

    obs_valid = valid[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']]
    obs_valid = obs_scaler.transform(obs_valid)
    tar_valid = np.asarray(valid['label_encoded'])
    tar_valid = tar_scaler.transform(tar_valid.reshape(-1, 1))

    obs_valid = pd.DataFrame(obs_valid)
    tar_valid = pd.DataFrame(tar_valid)
    transformed_valid = pd.concat([obs_valid, tar_valid], axis=1)
    transformed_valid.columns = columns
    transformed_valid['per_idx'] = valid['per_idx'].values

    obs_test = test[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']]
    obs_test = obs_scaler.transform(obs_test)
    tar_test = np.asarray(test['label_encoded'])
    tar_test = tar_scaler.transform(tar_test.reshape(-1, 1))

    obs_test = pd.DataFrame(obs_test)
    tar_test = pd.DataFrame(tar_test)
    transformed_test = pd.concat([obs_test, tar_test], axis=1)
    transformed_test.columns = columns
    transformed_test['per_idx'] = test['per_idx'].values

    X_train, y_train = dataProcessing(transformed_train, label_name='label_encoded', tw=100)
    X_valid, y_valid = dataProcessing(transformed_valid, label_name='label_encoded', tw=100)
    X_test, y_test = dataProcessing(transformed_test, label_name='label_encoded', tw=100)
    return X_train, y_train, X_valid, y_valid, X_test, y_test, obs_scaler, tar_scaler

def get_notch(dataset_dir):
    file_list = os.listdir(dataset_dir)
    csv_files = [file for file in file_list if file.endswith('.csv')]
    csv_files = sorted(csv_files)

    train = pd.DataFrame()
    for i in range(5):
        data = pd.read_csv(dataset_dir + csv_files[i])
        data['person_id'] = i + 1
        train = pd.concat([train, data])

    valid = pd.read_csv(dataset_dir + csv_files[5])
    valid['person_id'] = 6
    test = pd.read_csv(dataset_dir + csv_files[6])
    test['person_id'] = 7

    obs_scaler = StandardScaler()
    tar_scaler = StandardScaler()
    columns = ['acc_x', 'acc_y', 'acc_z', 'AnyFall', 'person_id']

    obs_train = train[['Acc_x [m/s^2]', 'Acc_y [m/s^2]', 'Acc_z [m/s^2]']]
    obs_train = obs_scaler.fit_transform(obs_train)
    tar_train = np.asarray(train['AnyFall'])
    tar_train = tar_scaler.fit_transform(tar_train.reshape(-1, 1))

    obs_train = pd.DataFrame(obs_train)
    tar_train = pd.DataFrame(tar_train)
    transformed_train = pd.concat([obs_train, tar_train], axis=1)
    transformed_train['per-id'] = train['person_id'].values
    transformed_train.columns = columns

    obs_valid = valid[['Acc_x [m/s^2]', 'Acc_y [m/s^2]', 'Acc_z [m/s^2]']]
    obs_valid = obs_scaler.transform(obs_valid)
    tar_valid = np.asarray(valid['AnyFall'])
    tar_valid = tar_scaler.transform(tar_valid.reshape(-1, 1))

    obs_valid = pd.DataFrame(obs_valid)
    tar_valid = pd.DataFrame(tar_valid)
    transformed_valid = pd.concat([obs_valid, tar_valid], axis=1)
    transformed_valid['per-id'] = valid['person_id'].values
    transformed_valid.columns = columns

    obs_test = test[['Acc_x [m/s^2]', 'Acc_y [m/s^2]', 'Acc_z [m/s^2]']]
    obs_test = obs_scaler.transform(obs_test)
    tar_test = np.asarray(test['AnyFall'])
    tar_test = tar_scaler.transform(tar_test.reshape(-1, 1))

    obs_test = pd.DataFrame(obs_test)
    tar_test = pd.DataFrame(tar_test)
    transformed_test = pd.concat([obs_test, tar_test], axis=1)
    transformed_test['per-id'] = test['person_id'].values
    transformed_test.columns = columns

    X_train, y_train = dataProcessing(transformed_train, 32, label_name='AnyFall', c_num=3)
    X_valid, y_valid = dataProcessing(transformed_valid, 32, label_name='AnyFall', c_num=3)
    X_test, y_test = dataProcessing(transformed_test, 32, label_name='AnyFall', c_num=3)
    return X_train, y_train, X_valid, y_valid, X_test, y_test, obs_scaler, tar_scaler

def get_smartfall(dataset_dir):
    train = pd.read_csv(dataset_dir + 'SmartFall Training.csv')
    test = pd.read_csv(dataset_dir + 'SmartFall Testing.csv')

    columns = ['acc_x', 'acc_y', 'acc_z', 'outcome']
    obs_scaler = StandardScaler()
    tar_scaler = StandardScaler()

    obs_train = train[[' ms_accelerometer_x', ' ms_accelerometer_y', ' ms_accelerometer_z']]
    obs_train = obs_scaler.fit_transform(obs_train)
    tar_train = np.asarray(train['outcome'])
    tar_train = tar_scaler.fit_transform(tar_train.reshape(-1, 1))

    obs_train = pd.DataFrame(obs_train)
    tar_train = pd.DataFrame(tar_train)
    transformed_train = pd.concat([obs_train, tar_train], axis=1)
    transformed_train.columns = columns
    transformed_train['per-id'] = 0

    obs_test = test[[' ms_accelerometer_x', ' ms_accelerometer_y', ' ms_accelerometer_z']]
    obs_test = obs_scaler.transform(obs_test)

    tar_test = np.asarray(test['outcome'])
    tar_test = tar_scaler.transform(tar_test.reshape(-1, 1))

    obs_test = pd.DataFrame(obs_test)
    tar_test = pd.DataFrame(tar_test)
    transformed_test = pd.concat([obs_test, tar_test], axis=1)
    transformed_test.columns = columns
    transformed_test['per-id'] = 0

    train, valid = train_test_split(transformed_train, test_size=0.25, shuffle=False, random_state=0)

    X_train, y_train = dataProcessing(train, tw=32, label_name='outcome', c_num=3)
    X_valid, y_valid = dataProcessing(valid, tw=32, label_name='outcome', c_num=3)
    X_test, y_test = dataProcessing(transformed_test, tw=32, label_name='outcome', c_num=3)
    return X_train, y_train, X_valid, y_valid, X_test, y_test, obs_scaler, tar_scaler

