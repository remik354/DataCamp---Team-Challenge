from pathlib import Path

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import pandas as pd
import os

DATA_PATH = Path('data')

def read_data(f_name):

    data = pd.read_csv(os.path.join('data/student+performance/student/', f_name), sep=";")

    return data



def merge_data():

    df_math = read_data('student-mat.csv')
    df_por = read_data('student-por.csv')

    merge_cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'schoolsup', 'famsup', 'activities' ,'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health']

    df_merged = df_math.merge(df_por, on=merge_cols, suffixes=("_math", "_por"))
    df_final = df_merged.drop(columns=["G1_por", "G2_por", "paid_por", "paid_math"])

    return df_final

def train_test():
    df = merge_data()



    df_sampled = df.sample(frac=0.2, random_state=42)
    train_df = df.drop(df_sampled.index)

    train_df.to_csv(DATA_PATH / 'train.csv', index=False)
    df_sampled.to_csv(DATA_PATH / 'test.csv', index=False)



if __name__ == '__main__':
    if not DATA_PATH.exists():
        DATA_PATH.mkdir()

    # Load the data
    print('Loading the data...', end='', flush=True)
    
    train_test()

    print('done')
