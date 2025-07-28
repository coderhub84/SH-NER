import pandas as pd
import ast
from sklearn.model_selection import train_test_split

def load_data(file_path, shuffle=True, seed=42):
    df = pd.read_csv(file_path)
    df['labels'] = df['labels'].apply(ast.literal_eval)
    df['words'] = df['text'].apply(lambda x: x.split())

    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df

def split_data(df, test_size=0.15, seed=42):
    return train_test_split(df, test_size=test_size, random_state=seed)

