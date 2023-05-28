
import pandas as pd

from datasets import load_dataset


def get_md_gender(config: str):
    source = load_dataset("md_gender_bias", config)['train']
    return pd.DataFrame.from_dict(source)
