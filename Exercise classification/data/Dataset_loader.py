import pandas
import torch
import numpy as np
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, df, tokenizer, labels):
        self.labels = [labels[label] for label in df['HOT_tag_xx']]
        # self.labels = [labels[label] for label in df['HOT_tag']]
        # self.labels = [labels[label] for label in df['Bloom_tag']]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=256,
                                truncation=True,
                                return_tensors="pt")
                      # for text in df['q_content']]
                      for text in df['q_content_en']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y
