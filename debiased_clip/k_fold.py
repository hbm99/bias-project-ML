import os
import random as rd

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def dataset_to_df(ds_path, k = 5):
    # Loading filenames
    for dirname, _, filenames in os.walk(ds_path):
        pass
    
    # Building the dataframe
    df = pd.DataFrame(filenames, columns = ['filename'] )
    df['filepath'] = df.filename.apply(lambda x: ds_path + x )
    df['age'] = df.filename.apply(lambda x: int(x.split('_')[0]))
    df['gender'] = df.filename.apply(lambda x: int(x.split('_')[1]))
    df['race'] = df.filename.apply(lambda x: int(x.split('_')[-2]))

    race_mapper = {0:'white', 1:'black',2: 'asian',3: 'indian', 4:'other'}
    gender_mapper = {0:'male',1:'female'}
    
    #Preprocess
    for i in range(len(df)):
        #img = np.array(Image.open(data_x[i]).resize((IMG_WIDTH,IMG_HEIGHT))) / 255.
        #images.append(img)

        #df['age']= df.age.apply(lambda x: x/116.)
        df['gender'][i]= gender_mapper[df['gender'][i]]
        df['race'][i]= race_mapper[df['race'][i]]
    

    #KFold
    skf=StratifiedKFold(n_splits=k, shuffle=True, random_state=1)

    X=df['filepath']
    y=df['gender']+ df['race']
    r= rd.randint(0,k-1)
    train_idx=[]
    test_idx=[]

    for i,(train_index, test_index) in enumerate(skf.split(X, y)):
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        if(i==r):
            train_idx=train_index
            test_idx=test_index
            break

    train_data= df.iloc[train_idx]
    test_data= df.iloc[test_idx]

    return train_data, test_data