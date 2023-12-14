import pandas as pd
import numpy as np

def load_data_set():
    df = pd.read_csv('.\\dataset\\Copper_Set.csv',dtype='unicode')
    return df

def masking_data(column,df):
    mask1 = df[column] <= 0
    print(mask1.sum())
    df.loc[mask1, column] = np.nan
    return df

def test_log_transforms(columns,df1):
    print('inside the method')
    for i in columns:
        df1[i] = np.log(df1[i])
    return df1


