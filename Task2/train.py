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

if __name__ == '__main__':
    config = get_config()
    # writer = SummaryWriter(f'./log/SOHU') 
    # save_dir = f"{config.model_dir}/ckpt"
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    
    # t = time.strftime("%Y-%m-%d-%-H:%M:%S", time.localtime())
    # save_model_dir = f"{save_dir}/{t}"
    # os.mkdir(save_model_dir)
    # os.popen(f'cp ./config.py {save_model_dir}/config.py')

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        print('GPU is ON!')
        device = torch.device(f'cuda')
    else:
        device = torch.device("cpu")
    # model = EmotionRecModel()
    # model = model.to(device)
    df_train_data = pd.read_csv('../baseline_recommend/data/train/sample.csv',sep=',')
    # df_valid_data = pd.read_csv("/work/yangshenghao/data/dataset/Sohu/evaluate/sample.csv",sep=',')
    # df_test_data = pd.read_csv("/work/yangshenghao/data/dataset/Sohu/submit/sample.csv",sep=',')
    df_train_path = '../baseline_recommend/data/train/sample.csv'
    item_emo_path = '../baseline_recommend/data/train/item_emotion.txt'
    item_emb_path = '../baseline_recommend/data/train/item_emb.pkl'
    print(f"df_train_data:{df_train_data}")
    # print(f"df_valid_data:{df_valid_data}")
    # print(f"df_test_data:{df_test_data}")
    train_dataloader = get_dataloader(df_train_path,item_emb_path,item_emo_path,config, shuffle=False)
    for train_step, batch_data in enumerate(tqdm(train_dataloader,desc=f"train progress")):
        user_id,item_id,city_id,label,item_emb,item_emo_feature,sequence_emb = \
            batch_data["user_id"],batch_data["item_id"],batch_data["city_id"],batch_data["label"],\
            batch_data["item_emb"],batch_data["item_emo_feature"],batch_data["sequence_emb"]
        # print(user_id,item_id,city_id,label,item_emb,item_emo_feature,sequence_emb)
        # print(hash_feature)
        