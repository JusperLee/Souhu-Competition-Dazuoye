###
# Author: Kai Li
# Date: 2022-04-14 12:04:57
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-04-14 15:22:19
###

import torch
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn
import torch.hub
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

class TrainDataset(Dataset):
    def __init__(self, datas, configs, is_train=True) -> None:
        super().__init__()
        self.is_train = is_train
        self.tokenizer = AutoTokenizer.from_pretrained(configs.model, cache_dir=os.path.join('./', configs.exp_name, "{}".format(configs.model.split('/')[-1])))
        special_tokens_dict = {'additional_special_tokens': ['[AS]','[AE]']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.datas = datas
        self.max_len = 512
        
    def __len__(self,):
        return len(self.datas)
    
    def __getitem__(self, index):
        x = self.datas[index]
        sentence=x.sentence
        sentence=sentence[:x.aspect_from_pos+len(sentence)-len(sentence)]+'[AS]{}[AE]'.format(x.aspect_term)+x.sentence[x.aspect_from_pos+len(x.aspect_term):]
        sentence_token = self.tokenizer(sentence)
        new_sentence = self.tokenizer.decode(sentence_token['input_ids'])
        new_sentence = new_sentence.split(' ')
        idx = new_sentence.index('[AS]')
        label = x.polarity
        if len(sentence_token['input_ids']) < self.max_len:
            sentence_token['input_ids'] += ([0] * (self.max_len - len(sentence_token['input_ids'])))
            sentence_token['attention_mask'] += ([0] * (self.max_len - len(sentence_token['attention_mask'])))
        if len(sentence_token['input_ids']) > self.max_len:
            sentence_token['input_ids'] = sentence_token['input_ids'][:self.max_len]
            sentence_token['attention_mask'] = sentence_token['attention_mask'][:self.max_len]
        return {"input_ids": torch.from_numpy(np.array(sentence_token['input_ids'], dtype='int64')), 
                "attention_mask": torch.from_numpy(np.array(sentence_token['attention_mask'], dtype='int64')), 
                "label": torch.from_numpy(np.array([label], dtype='int64')),
                "idx": torch.from_numpy(np.array([idx], dtype='int64'))}