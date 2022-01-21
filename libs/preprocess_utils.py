import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder

def preprocess_mobi_person_activity_info(dataset_dir):
    readme_file = open(dataset_dir + 'Readme.txt', 'r', encoding='latin1')
    strings = readme_file.readlines()

    person_list, activity_list = [], []
    for s in strings:
        if 'sub' in s and '|' in s:
            temp = s.split('|')
            temp = [x.strip() for x in temp]
            if len(temp) == 9:
                person_list.append(temp[3:-1])
        if '|' in s:
            temp = s.split('|')
            temp = [x.strip() for x in temp]
            if len(temp) == 0:
                activity_list.append(temp[1:-1])

    falls = ['FOL', 'FKL', 'BSC', 'SDL']
    columns = ['name', 'age', 'height', 'weight', 'gender']

    person_info = pd.DataFrame(person_list, columns=columns)

    activity_info = pd.DataFrame(activity_list)
    activity_info.columns = activity_info.iloc[0]
    activity_info = activity_info.drop(0)
    activity_info = activity_info.drop(13)
    activity_info = activity_info.reset_index(drop=True)
    index = activity_info['No.']
    activity_info = activity_info.drop(['No.'], axis=1)
    activity_info.index = index
    activity_info['label_encoded'] = list(range(len(activity_info)))
    return person_info, activity_info

def preprocess_mobi_df(dataset_dir, preprocessed_dataset_dir):
    person_info, activity_info = preprocess_mobi_person_activity_info(dataset_dir=dataset_dir)

    train = pd.read_csv(f'{preprocessed_dataset_dir}/train.csv', index_col=0)
    valid = pd.read_csv(f'{preprocessed_dataset_dir}/valid.csv', index_col=0)
    test = pd.read_csv(f'{preprocessed_dataset_dir}/test.csv', index_col=0)

    encoder = LabelEncoder()
    encoder.fit(activity_info['Label'])

    train_encoded = encoder.transform(train['label'])
    train['label_encoded'] = train_encoded

    valid_encoded = encoder.transform(valid['label'])
    valid['label_encoded'] = valid_encoded

    test_encoded = encoder.transform(test['label'])
    test['label_encoded'] = test_encoded
    return train, valid, test