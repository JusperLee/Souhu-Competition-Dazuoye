from torch.utils.data import Dataset, DataLoader
import random
from tensorflow import feature_column as fc
import tensorflow as tf
import torch
import pandas as pd 
import pickle as pk
from tqdm import tqdm
import numpy as np


class TrainDataset(Dataset):
    def __init__(self,user_ids_hash,item_ids_hash,citys_hash,sequence_list,
    item_ids, item_embs,item_emo_feature_dict,item2vecs,labels=None,train=True):
        self.user_ids_hash = user_ids_hash
        self.item_ids_hash = item_ids_hash
        self.citys_hash = citys_hash
        self.sequence_list = sequence_list
        self.item_ids = item_ids
        self.labels = labels
        self.train = train
        self.item_embs = item_embs
        self.item_emo_feature_dict = item_emo_feature_dict
        self.item2vecs = item2vecs
    def __len__(self):
        return len(self.user_ids_hash)
    
    def __getitem__(self, item):
        # print(item)
        user_id_hash = self.user_ids_hash[item]
        item_id_hash = self.item_ids_hash[item]
        city_id_hash = self.citys_hash[item]
        sequence = self.sequence_list[item]
        item_id = self.item_ids[item]
        item2vec = self.item2vecs[int(item_id)-10000000]
        item_emb = self.item_embs[int(item_id)-10000000]
        item_emo_feature = self.item_emo_feature_dict[item_id]
        sequence_emb_list_emo = []
        item2vec_seq_list = []
        item_emb_seq = []
        max_seq_len = 21 # max length of sequence
        item_emb_seq = -1*torch.ones(max_seq_len)
        if len(sequence) == 0:
            sequence_emb_emo = np.zeros((1,768*3))
            item2vec_seq = np.zeros((1,100))
        else:
            for i, sequence_item_id in enumerate(sequence):
                sequence_emb_list_emo.append(self.item_embs[int(sequence_item_id)-10000000]) # n_items * 768
                item2vec_seq_list.append(self.item2vecs[int(sequence_item_id)-10000000])
                if i < max_seq_len:
                    item_emb_seq[i] = self.item_ids_hash[int(sequence_item_id)-10000000]
            sequence_emb_emo = np.mean(np.array(sequence_emb_list_emo),axis=0)# 1 * 768 
            sequence_emb_emo = np.reshape(sequence_emb_emo, (1,-1))
            item2vec_seq = np.mean(np.array(item2vec_seq_list),axis=0)# 1 * 100
            item2vec_seq = np.reshape(item2vec_seq, (1,-1))
        item_emb_seq = item_emb_seq.long()
        if self.train:
            label = self.labels[item]
        else:
            label = -1
        return{
                "user_id":user_id_hash,
                "item_id":item_id_hash,
                "city_id":city_id_hash,
                "label": label,
                "item_emb":item_emb,
                "item_emo_feature":item_emo_feature,
                "sequence_emb":sequence_emb_emo.flatten(),
                'item_emb_seq':item_emb_seq,
                'item2vec':item2vec,
                'item2vec_seq':item2vec_seq
            }


def get_dataloader(path,item_emb_path,item_emo_path,config,train=True, shuffle=True):
    df_data = pd.read_csv(path)
    with open(item_emb_path,'rb') as f:
        item_embs = pk.load(f)
    with open(item_emo_path,'r',encoding="utf-8") as f:
        lines = f.readlines()[1:]
    item2vec = torch.load(config.item2vec_path).cpu().detach().numpy()
    item_emo_feature_dict = {}

    # emotion features
    item_id_count = 10000000
    for line in lines:
        item_id, item_dict = line.strip().split("\t")
        item_dict = item_dict.strip('"').replace('""','"')
        item_dict = eval(item_dict)
        emotions = np.array(list(item_dict.values()))+3
        while int(item_id) != item_id_count:
            item_emo_feature_dict[str(item_id_count)] = torch.tensor([0,0,0,0]).float()
            item_id_count += 1
        if len(emotions) == 0:
            item_emo_feature_dict[item_id] = torch.tensor([0,0,0,0]).float()
            continue
        max_emo = np.max(emotions)
        min_emo = np.min(emotions)
        mean_emo = np.mean(emotions)
        std_emo = np.std(emotions)
        item_emo_feature_dict[item_id] = torch.tensor([max_emo,min_emo,mean_emo,std_emo]).float()
        item_id_count += 1

    # user and item features
    user_ids = df_data['suv'].values.tolist()
    item_ids = df_data['itemId'].values.tolist()
    item_ids = [str(x) for x in item_ids]
    citys = df_data['city'].values.tolist()
    citys = [str(x) for x in citys]

    # user sequence features
    sequences = df_data['userSeq'].values.tolist()
    sequence_list = []
    for sequence in sequences:
        item_list = []
        if pd.isna(sequence):
            sequence_list.append(item_list)
            continue
        example_list = sequence.split(';')
        for example in example_list:
            item = example.split(':')[0]
            item_list.append(item)
        sequence_list.append(item_list)
    # print(f"sequence_list: {len(sequence_list)}")
    with tf.device('/cpu:0'):
        user_ids_hash = torch.tensor(tf.strings.to_hash_bucket_strong(user_ids, 300000, [1,2]).numpy())
        item_ids_hash = torch.tensor(tf.strings.to_hash_bucket_strong(item_ids, 4000, [1,2]).numpy())
        citys_hash = torch.tensor(tf.strings.to_hash_bucket_strong(citys, 340, [1,2]).numpy())
    if train:
        labels = df_data['label'].values.tolist()
    else:
        labels = None

    ds = TrainDataset(user_ids_hash,item_ids_hash,citys_hash,sequence_list,
    item_ids,item_embs,item_emo_feature_dict,item2vec,labels,train)
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        num_workers=8,
        shuffle=shuffle,
    )