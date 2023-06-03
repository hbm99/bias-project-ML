

from typing import List

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", DEVICE)

def run_clip(attributes: List[str], labels: List[List[str]], tkns: List[List[str]], df: pd.DataFrame, model = CLIP_MODEL) -> pd.DataFrame:
    texts = [clip.tokenize(tkns[i]).to(DEVICE) for i in range(len(tkns))]
    photos = [Image.open(photo_path) for photo_path in df['filepath']]
    
    results = {attribute : [] for attribute in attributes}
    pending_photos = len(photos)
    for i in range(0, len(photos), min(BATCH_SIZE, pending_photos)):
        pending_photos = len(photos) - i
        images = [CLIP_PREPROCESS(photos[photo_idx]) for photo_idx in range(i, min(i + BATCH_SIZE, len(photos)))]
        image_input = torch.tensor(np.stack(images)).to(DEVICE)
        with torch.no_grad():
            logits_per_image_list = [model(image_input, text)[0] for text in texts]
            
            probs_list = [logits_per_image_list[j].softmax(dim=-1).cpu().numpy() for j in range(len(logits_per_image_list))]
            
            for k in range(len(attributes)):
                for j in range(len(probs_list[k])):
                    results[attributes[k]].append(probs_list[k][j])
                
    predictions = [np.argmax(results[attribute], axis = 1) for attribute in attributes]
    
    for i in range(len(labels)):
        get_label = lambda x: labels[i][x]
        vect_get_label = np.vectorize(get_label)
        df['predicted_' + attributes[i]] = vect_get_label(predictions[i])
    
    return df