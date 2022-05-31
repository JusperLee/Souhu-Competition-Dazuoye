from torch.utils.data import Dataset, DataLoader
import random
from tensorflow import feature_column as fc
import tensorflow as tf
import torch
import pandas as pd 
if tf.__version__ >= '2.0':
  tf = tf.compat.v1
class TrainDataset(Dataset):
    def __init__(self,user_ids,item_ids,citys,config,labels=None,train=True):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.citys = citys
        self.user_cate = fc.indicator_column(fc.categorical_column_with_hash_bucket("suv", 40000))
        self.feed_cate = fc.indicator_column(fc.categorical_column_with_hash_bucket("itemId", 4000, tf.int64))
        self.city_cate = fc.indicator_column(fc.categorical_column_with_hash_bucket("city", 200, tf.int64))
        self.columns = [self.user_cate,self.feed_cate,self.city_cate]
        self.labels = labels
        self.train = train
    def __len__(self):
        return len(self.user_ids)
    def __getitem__(self, item):
        user_ids = self.user_ids[item]
        item_ids = self.item_ids[item]
        citys = self.citys[item]
        hash_dict = {'suv':[user_ids],'itemId':[item_ids],'city':[citys]}
        hash_feature = tf.feature_column.input_layer(hash_dict, self.columns).numpy().flatten()
        city_id, item_id, user_id = torch.nonzero(torch.tensor(hash_feature), as_tuple=True)[0]
        user_id -= 4200
        item_id -= 200
        if self.train:
            label = self.labels[item]
        else:
            label = None
        return{
                "user_id":user_id,
                "item_id":item_id,
                "city_id":city_id,
                "label": label
            }

def get_dataloader(path,config,train=True):
    df_data = pd.read_csv(path)
    user_ids = df_data['suv'].values.tolist()
    item_ids = df_data['itemId'].values.tolist()
    citys = df_data['city'].values.tolist()
    if train:
        labels = df_data['label'].values.tolist()
    else:
        labels = None
    ds = TrainDataset(user_ids,item_ids,citys,config,labels,train)
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        num_workers=40,
        shuffle=True,
    )