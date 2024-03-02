import pandas as pd
import torch
import numpy as np
from collections import Counter


def bloom_2_hot(df, Bloom_list, Hot_list):
    for i in range(len(df)):
        df['HOT_tag'][i] = Hot_list[df['Bloom_tag'][i]]
    print(df['HOT_tag'].value_counts())
    # df.to_excel('q_list_en_2.xlsx', index=False)


def smote(df):
    labels = df['HOT_tag_xx']
    print(Counter(labels))


if __name__ == '__main__':
    df = pd.read_excel('C:/Users/94207/OneDrive/Files/4-code/Datasets/20231129/q_list.xlsx')
    # Bloom_list = df['Bloom_tag'].unique()
    # Hot_list = {'Applying': '0', 'Creating': '1', 'Analyzing': '1',
    #             'Evaluating': '1', 'Remembering': '0', 'Understanding': '0'}
    smote(df)


