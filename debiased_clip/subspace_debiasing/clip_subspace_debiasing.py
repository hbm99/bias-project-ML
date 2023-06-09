from __future__ import annotations

import sys
from typing import List

import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

sys.path.append('/Users/hanselblanco/Documents/4to/ML/project/bias-project-ML')
import clip
import numpy as np
import pandas as pd
from PIL import Image

from data.md_gender.md_gender import get_md_gender
from data_preprocess import data_selection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
R_COUNT = 200
CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", DEVICE)
BATCH_SIZE = 4

def get_data_to_encode(path):
    # Load dataframes
    df_pictures, df_test_pictures = data_selection(path)
    df_text = get_md_gender('image_chat')

    df_pictures_males = df_pictures.loc[df_pictures['gender'] == 'male'].sample(R_COUNT)
    df_pictures_females = df_pictures.loc[df_pictures['gender'] == 'female'].sample(R_COUNT)
    
    df_text_males = df_text.loc[df_text['male'].values & ~ df_text['female'].values].sample(R_COUNT)
    df_text_females = df_text.loc[df_text['female'].values & ~ df_text['male'].values].sample(R_COUNT)
    
    return [df_pictures_males, df_pictures_females], [df_text_males, df_text_females], df_test_pictures

def encode(pic_dataframes: List[pd.Dataframe], text_dataframes: List[pd.Dataframe]):
    for df in pic_dataframes:
        df['image_tensor'] = df.filepath.apply(lambda x: torch.tensor(np.stack([CLIP_PREPROCESS(Image.open(x))])).cpu())
        df['image_encoded'] = df.image_tensor.apply(lambda x: CLIP_MODEL.encode_image(x).float())
    for df in text_dataframes:
        df['text_tensor'] = df.caption.apply(lambda x: clip.tokenize(x).cpu())
        df['text_encoded'] = df.text_tensor.apply(lambda x: CLIP_MODEL.encode_text(x).float())
    return pic_dataframes, text_dataframes

def get_gender_subspace(R_male: pd.Series, R_female: pd.Series, k: int | float = 164):
    mean_male = R_male.sum() / (R_COUNT * 2)
    mean_female = R_female.sum() / (R_COUNT * 2)
    
    R_male = R_male.apply(lambda x: x - mean_male)
    R_female = R_female.apply(lambda x: x - mean_female)
    
    R = pd.concat([R_male, R_female])
    
    pca = PCA(k)
    R = R.apply(lambda x: x.detach().numpy()[0]).tolist()
    pca.fit(R)
    
    # plot_variance_ratio(pca)
    
    return pca.components_

def plot_variance_ratio(pca):
    exp_var = pca.explained_variance_ratio_ * 100
    cum_exp_var = np.cumsum(exp_var)

    plt.bar(range(1, len(pca.components_) + 1), exp_var, align='center',
            label='Individual explained variance')

    plt.step(range(1, len(pca.components_) + 1), cum_exp_var, where='mid',
            label='Cumulative explained variance', color='red')

    plt.ylabel('Explained variance percentage')
    plt.xlabel('Principal component index')
    plt.xticks(ticks=[i for i in range(0, 512, 50)])
    plt.legend(loc='best')
    plt.tight_layout()
    
    plt.gcf().set_size_inches(64, 48)
    plt.savefig('variance_ratio.png', dpi=256)

def run_clip_gender_debiased(labels: List[str], tkns: List[str], df: pd.DataFrame, transform_matrix, alpha: float) -> pd.DataFrame:
    texts = clip.tokenize(tkns).cpu()
    photos = [Image.open(photo_path) for photo_path in df['filepath']]
    
    with torch.no_grad():
        text_features = get_gender_tensor(texts, CLIP_MODEL.encode_text, transform_matrix,alpha, debiased = True)
    text_features = text_features / text_features.norm(dim = -1, keepdim = True)
    results = []
    pending_photos = len(photos)
    for i in range(0, len(photos), min(BATCH_SIZE, pending_photos)):
        pending_photos = len(photos) - i
        images = [CLIP_PREPROCESS(photos[photo_idx]) for photo_idx in range(i, min(i + BATCH_SIZE, len(photos)))]
        image_input = torch.tensor(np.stack(images)).to(DEVICE)
        with torch.no_grad():
            image_features = get_gender_tensor(image_input, CLIP_MODEL.encode_image, transform_matrix,alpha, debiased = True)
        
        image_features = image_features / image_features.norm(dim = -1, keepdim = True)
        
        # cosine similarity as logits
        probs = (100.0 * text_features @ image_features.T).softmax(dim=-1) 
        
        results.append(probs) 
    
    flatten_results = torch.cat(results, axis=1)
    predictions = torch.argmax(flatten_results, axis=0)
    
    get_label = lambda x:labels[x]
    vgetlabel = np.vectorize(get_label)
    genders = vgetlabel(predictions)
    
    df['predicted_gender'] = genders
    
    return df

def get_gender_tensor(input, encoder, transform_matrix, alpha: float, debiased: bool = True):
    H = encoder(input).float() 
    if not debiased:
        return H
    mult = H @ torch.from_numpy(transform_matrix).float()
    diff = H - (mult * alpha)
    return diff
    
def subspace_debiasing(path: str, alpha: float = 0.4):
    dfs_pics, dfs_text, df_test_pics = get_data_to_encode(path)
    dfs_pics, dfs_text = encode(dfs_pics, dfs_text)
    R_male = dfs_pics[0]['image_encoded'].append(dfs_text[0]['text_encoded'], ignore_index=True)
    R_female = dfs_pics[1]['image_encoded'].append(dfs_text[1]['text_encoded'], ignore_index=True)
    V = get_gender_subspace(R_male, R_female)
    transform_matrix = np.dot(np.transpose(V), V)
    labels = ['male', 'female']
    df_test_pics = run_clip_gender_debiased(labels, 
                                            ['A person of gender ' + label for label in labels],
                                            df_test_pics,
                                            transform_matrix, alpha)
    return df_test_pics

