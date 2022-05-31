import os
import random
import numpy as np
import torch
import time
import pandas as pd
from tqdm import tqdm
# from model import EmotionRecModel
from config import get_config
from dataloader_new import get_dataloader
from torch.utils.tensorboard import SummaryWriter
from model import *
import tensorflow as tf 

if __name__ == '__main__':
    config = get_config()
    train_dataloader = get_dataloader('../baseline_recommend/data/evaluate/sample.csv',config,train=True)
    test_data = next(iter(train_dataloader))
    user_hash = test_data['user_id'].cuda()
    item_hash = test_data['item_id'].cuda()
    city_hash = test_data['city_id'].cuda()
    test_embedding = rec_embedding().cuda()
    import pdb; pdb.set_trace()
    out = test_embedding(user_hash, item_hash, city_hash)
