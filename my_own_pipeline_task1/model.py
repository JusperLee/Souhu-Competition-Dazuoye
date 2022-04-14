###
# Author: Kai Li
# Date: 2022-04-14 11:50:01
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-04-14 13:45:08
###
from distutils.command.config import config
import os
import torch
from turtle import pd
import numpy as np
import torch.nn
import torch.hub
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoConfig

class RobertaATSA(torch.nn.Module):
    def __init__(self, configs):
        super(RobertaATSA, self).__init__()

        self.configs=configs
        self.pretrain_config = AutoConfig.from_pretrained(configs.model)
        self.roberta=AutoModel.from_pretrained(configs.model, cache_dir=os.path.join('./', configs.exp_name, "{}".format(configs.model.split('/')[-1])))
        self.tokenizer = AutoTokenizer.from_pretrained(configs.model, cache_dir=os.path.join('./', configs.exp_name, "{}".format(configs.model.split('/')[-1])))
        
        special_tokens_dict = {'additional_special_tokens': ['[AS]','[AE]']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.roberta.resize_token_embeddings(len(self.tokenizer))
        
        self.linear_hidden=torch.nn.Linear(configs.ROBERTA_DIM,configs.LINEAR_HIDDEN_DIM)
        
        self.linear_output=torch.nn.Linear(configs.LINEAR_HIDDEN_DIM,5)

        self.dropout_output=torch.nn.Dropout(0.1)
        self._init_weights(self.linear_hidden)
        self._init_weights(self.linear_output)
        
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.pretrain_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.pretrain_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def loss(self,logits,labels):
        loss_fnc = torch.nn.CrossEntropyLoss()
        # loss_fnc = DiceLoss(smooth = 1, square_denominator = True, with_logits = True,  alpha = 0.01 )
        loss = loss_fnc(logits, labels)
        return loss
    
    def forward(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        idx = batch['idx'].reshape(-1)
        features = self.roberta(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        atsa_features = features.last_hidden_state[0, idx]
        atsa_features=self.linear_hidden(atsa_features)
        atsa_features=F.gelu(atsa_features)
        predictions=self.linear_output(atsa_features)
        return predictions, self.loss(predictions, batch['label'].reshape(-1))