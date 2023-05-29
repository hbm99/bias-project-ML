from __future__ import annotations

import sys
from typing import List

import torch
from sklearn.decomposition import PCA

sys.path.append('/Users/hanselblanco/Documents/4to/ML/project/bias-project-ML')
import clip
import numpy as np
import pandas as pd
from PIL import Image

from data.md_gender.md_gender import get_md_gender
from data_preprocess import data_selection

R_COUNT = 100
CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", "cuda" if torch.cuda.is_available() else "cpu")

def get_data_to_encode():
    # Load dataframes
    df_pictures, df_test_pictures = data_selection()
    df_text = get_md_gender('image_chat')

    df_pictures_males = df_pictures.loc[df_pictures['gender'] == 'male'].sample(R_COUNT)
    df_pictures_females = df_pictures.loc[df_pictures['gender'] == 'female'].sample(R_COUNT)
    
    # a = df_text['male'].values and not df_text['female'].values
    # print(a)
    df_text_males = df_text.loc[df_text['male'].values & ~ df_text['female'].values].sample(R_COUNT)
    df_text_females = df_text.loc[df_text['female'].values & ~ df_text['male'].values].sample(R_COUNT)
    
    return [df_pictures_males, df_pictures_females], [df_text_males, df_text_females], df_test_pictures

def encode(pic_dataframes: List[pd.Dataframe], text_dataframes: List[pd.Dataframe]):
    for df in pic_dataframes:
        df['image_tensor'] = df.filepath.apply(lambda x: torch.tensor(np.stack([CLIP_PREPROCESS(Image.open(x))])))
        df['image_encoded'] = df.image_tensor.apply(lambda x: CLIP_MODEL.encode_image(x))
    for df in text_dataframes:
        df['text_tensor'] = df.caption.apply(lambda x: clip.tokenize(x).to(device = 'cpu'))
        df['text_encoded'] = df.text_tensor.apply(lambda x: CLIP_MODEL.encode_text(x))
    return pic_dataframes, text_dataframes

def get_gender_subspace(R_male: pd.Series, R_female: pd.Series, k: int = 5):
    mean_male = R_male.sum() / R_COUNT
    mean_female = R_female.sum() / R_COUNT
    
    R_male = R_male.apply(lambda x: x - mean_male)
    R_female = R_female.apply(lambda x: x - mean_female)
    
    R = pd.concat([R_male, R_female])
    
    pca = PCA(n_components = k)
    pca.fit(R)
    
    return pca.components_
    

if __name__ == '__main__':
    dfs_pics, dfs_text, df_test_pics = get_data_to_encode()
    dfs_pics, dfs_text  = encode(dfs_pics, dfs_text)
    R_male = dfs_pics[0]['image_encoded'].append(dfs_text[0]['text_encoded'], ignore_index=True)
    R_female = dfs_pics[1]['image_encoded'].append(dfs_text[1]['text_encoded'], ignore_index=True)
    V = get_gender_subspace(R_male, R_female)
    
    
    
    
    
    


