
import pandas as pd

# Convert JSON to DataFrame
df = pd.read_json('test_data.json')
print(df)


get_new_path = lambda x : '/Users/hanselblanco/Documents/4to/ML/project/bias-project-ML/data/utkface/' + x.split('/')[-1]

df['filepath'] = df['filepath'].apply(get_new_path)

print(df['filepath'])

df.to_json('test_data.json')

