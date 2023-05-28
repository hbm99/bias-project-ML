from __future__ import annotations

import sys
from typing import List

import torch

sys.path.append('/Users/hanselblanco/Documents/4to/ML/project/bias-project-ML')
import clip
import pandas as pd
from PIL import Image

from data.md_gender.md_gender import get_md_gender
from data_preprocess import data_selection

R_COUNT = 1000
CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", "cuda" if torch.cuda.is_available() else "cpu")

def get_data_to_encode():
    # Load dataframes
    df_pictures, df_test_pictures = data_selection()
    df_text = get_md_gender('image_chat')

    df_pictures_males = df_pictures.loc[df_pictures['gender'] == 'male'].sample(R_COUNT)
    df_pictures_females = df_pictures.loc[df_pictures['gender'] == 'female'].sample(R_COUNT)

    df_text_males = df_text.loc[df_text['male'] and not df_text['female']].sample(R_COUNT)
    df_text_females = df_text.loc[df_text['female'] and not df_text['male']].sample(R_COUNT)
    
    return [df_pictures_males, df_pictures_females], [df_text_males, df_text_females]

def encode(pic_dataframes: List[pd.Dataframe], text_dataframes: List[pd.Dataframe]):
    for df in pic_dataframes:
        df['image_tensor'] = df.photo_path.apply(lambda x: torch.tensor(CLIP_PREPROCESS(Image.open(x))))
        df['image_encoded'] = df.image_tensor.apply(lambda x: CLIP_MODEL.encode_image(x))
    for df in text_dataframes:
        df['text_encoded'] = df.text.apply(lambda x: CLIP_MODEL.encode_text(x))
    return pic_dataframes, text_dataframes

def get_gender_subspace(R_male: pd.Series, R_female: pd.Series):
    pass

if __name__ == '__main__':
    dfs_pics, dfs_text = encode(get_data_to_encode())
    R_male = dfs_pics[0]['image_encoded'].append(dfs_text[0]['text_encoded'], ignore_index=True)
    R_female = dfs_pics[1]['image_encoded'].append(dfs_text[1]['text_encoded'], ignore_index=True)
    V = get_gender_subspace(R_male, R_female)
    
    
    
    


