import os
import random as rd

import pandas as pd
from sklearn.model_selection import StratifiedKFold

RACE_MAPPER = {0:'white', 1:'black',2: 'asian',3: 'indian', 4:'other'}
GENDER_MAPPER = {0:'male',1:'female'}


def data_selection(ds_path: str = 'data/utkface', k: int = 5):
    
    df = load_dataset(ds_path)
    
    df = map_values(df)
    
    # Stratified KFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)

    X = df['filepath']
    y = df['gender'] + df['race']
    r = rd.randint(0, k - 1)
    train_idx = []
    test_idx = []

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        if i == r:
            train_idx=train_index
            test_idx=test_index
            break

    train_data = df.iloc[train_idx]
    test_data = df.iloc[test_idx]

    return train_data, test_data

def load_dataset(ds_path: str = 'data/utkface'):
    # Loading filenames
    filenames = os.listdir(ds_path)
    
    try:
        filenames.remove('.DS_Store')
    except:
        pass
    
    # Building the dataframe
    df = pd.DataFrame(filenames, columns = ['filename'] )
    df['filepath'] = df.filename.apply(lambda x: ds_path + x )
    df['age'] = df.filename.apply(lambda x: int(x.split('_')[0]))
    df['gender'] = df.filename.apply(lambda x: int(x.split('_')[1]))
    df['race'] = df.filename.apply(lambda x: int(x.split('_')[-2]))
    
    return df

def map_values(df: pd.DataFrame):
    for i in range(len(df)):
        df['gender'][i]= GENDER_MAPPER[df['gender'][i]]
        df['race'][i]= RACE_MAPPER[df['race'][i]]
    return df
    

