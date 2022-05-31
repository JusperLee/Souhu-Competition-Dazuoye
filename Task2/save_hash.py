#%%
from torch.utils.data import Dataset, DataLoader
import random
from tensorflow import feature_column as fc
import tensorflow as tf
import torch
import pandas as pd 

path = '../baseline_recommend/data/evaluate/sample.csv'

df_data = pd.read_csv(path)
user_ids = df_data['suv'].values.tolist()
item_ids = df_data['itemId'].values.tolist()
item_ids = [str(x) for x in item_ids]
citys = df_data['city'].values.tolist()
citys = [str(x) for x in citys]
with tf.device('/cpu:0'):
    user_ids = torch.tensor(tf.strings.to_hash_bucket(user_ids, 40000).numpy())
    item_ids = torch.tensor(tf.strings.to_hash_bucket(item_ids, 4000).numpy())
    citys = torch.tensor(tf.strings.to_hash_bucket(citys, 200).numpy())
# %%
