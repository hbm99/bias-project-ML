
import pandas as pd

from datasets import load_dataset


def image_chat_md_gender():
    source = load_dataset("md_gender_bias", "image_chat")['train']
    return pd.DataFrame.from_dict(source)

print(image_chat_md_gender())