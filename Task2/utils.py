import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import pickle as pk
from torch.utils.data import Dataset, DataLoader
import time
import evaluation
from tqdm import tqdm
import tensorflow as tf
if tf.__version__ >= '2.0':
  tf = tf.compat.v1
from tensorflow import feature_column as fc
from params import *


def file_to_dataset(file_path, isTest, shuffle, batch_size, num_epochs):
    if (isTest):
        label_name = None
    else:
        label_name = "label"
    ds = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=batch_size,
        label_name=label_name,
        shuffle = shuffle,
        shuffle_seed = SEED,
        num_epochs = num_epochs,
        prefetch_buffer_size = tf.data.experimental.AUTOTUNE,

    )
    return ds


def get_feature_columns():
    feature_columns = list()
    user_cate = fc.categorical_column_with_hash_bucket("suv", 40000)
    feed_cate = fc.categorical_column_with_hash_bucket("itemId", 4000, tf.int64)
    city_cate = fc.categorical_column_with_hash_bucket("city", 200, tf.int64)
    user_embedding = fc.embedding_column(user_cate, EMBED_DIM, max_norm=EMBED_L2)
    feed_embedding = fc.embedding_column(feed_cate, EMBED_DIM, max_norm=EMBED_L2)
    city_cate_embedding = fc.embedding_column(city_cate, EMBED_DIM, max_norm=EMBED_L2)
    feature_columns.append(user_embedding)
    feature_columns.append(feed_embedding)
    feature_columns.append(city_cate_embedding)
    return feature_columns


class sougou_dataset(Dataset):
    def __init__(self, fc, label):
        super(sougou_dataset, self).__init__()
        self.fc = fc
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.fc[item], self.label[item]



class sougou_dataset_tf():
    def __init__(self, file_path, batch_sz, isTest=False):
        super(sougou_dataset_tf, self).__init__()
        if isTest:
            batch_sz *= 10
        self.ds = file_to_dataset(file_path, isTest=isTest, shuffle=False, batch_size=batch_sz,
                                   num_epochs=1)
        self.ds_list = list(self.ds.as_numpy_iterator())
        self.feature_columns = get_feature_columns()

    def get(self, idx):
        x = self.ds_list[idx]
        return torch.tensor(tf.feature_column.input_layer(x[0], self.feature_columns))


