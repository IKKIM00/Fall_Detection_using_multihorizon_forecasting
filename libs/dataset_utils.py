import pandas as pd
from sklearn.preprocessing import LabelEncoder

def open_mobiact(dataset_dir):
    activity_info = mobiact_collect_actinfo()

    train = pd.read_csv(dataset_dir + 'train.csv', index_col=0)
    valid = pd.read_csv(dataset_dir + 'valid.csv', index_col=0)
    test = pd.read_csv(dataset_dir + 'test.csv', index_col=0)

    label_encoder = LabelEncoder()
    label_encoder.fit(activity_info['Label'])



def mobiact_collect_actinfo(dataset_dir='dataset/mobiact_dataset'):
    """
    params:
        dataset_dir - dir of mobiact_dataset
    return:
        preprocessed_mobiact_dataset
    """

    # 1. collect information included in readme.txt file
    txt_file = open(dataset_dir + 'Readme.txt', 'r', encoding='latin1')
    txt_content = txt_file.readlines()
    txt_file.close()

    activity_list = []
    for s in txt_content:
        if '|' in s:
            temp = [x.strip() for x in s.split('|')]
            if len(temp) == 8:
                activity_list.append(temp[1:-1])

    # 2. create activity info dataframe
    activity_info = pd.DataFrame(activity_list)
    activity_info.columns = activity_info.iloc[0]
    activity_info = activity_info.drop([0, 13])
    activity_info = activity_info.reset_index(drop=True)
    activity_info.index = activity_info['No.']
    activity_info.drop(['No.'], axis=1)
    return activity_info