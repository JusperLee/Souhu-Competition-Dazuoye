import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from item2vec_model import Item2Vec
from tqdm import tqdm

args = {
    'context_window': 2,
    'vocabulary_size': 146790, # df['item'].nunique(),
    'rho': 1e-5,  # threshold to discard word in a sequence
    'batch_size': 256*6,
    'embedding_dim': 100,
    'epochs': 5,
    'learning_rate': 0.001,
    "data_path":"train_data.tsv",
}

model = Item2Vec(args)
model.fit()
