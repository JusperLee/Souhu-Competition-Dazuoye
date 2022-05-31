from sched import scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import os
import pandas as pd
import numpy as np
import pickle as pk
from utils import *
from model import *
from params import *
from dataloader_new import *
from config import get_config
from tqdm import tqdm
import tensorflow as tf
if tf.__version__ >= '2.0':
  tf = tf.compat.v1




def evaluate(model, df, eval_dataloader):
    predicts = []
    labels = []
    pvid_list = df['pvId'].astype(str).tolist()
    for data in tqdm(eval_dataloader):
        user_hash = data['user_id'].to(device)
        item_hash = data['item_id'].to(device)
        city_hash = data['city_id'].to(device)
        item_emb = data["item_emb"].to(device)
        item_emb_seq = data['item_emb_seq'].to(device)
        item_emo_feature = data["item_emo_feature"].to(device)
        sequence_emb = data["sequence_emb"].to(device)
        item2vec = data["item2vec"].to(device)
        label = data['label'].detach().cpu().numpy().flatten()
        predicts.append(F.softmax(
            model(user_hash, item_hash, city_hash, item_emb, item_emo_feature, sequence_emb, item_emb_seq, item2vec) \
                .detach().cpu()).numpy()[:,1].flatten())
        labels.append(label)
    predicts = np.concatenate(predicts)
    labels = np.concatenate(labels)
    gauc = evaluation.gAUC(labels, predicts, pvid_list)
    return gauc


def predict(submit_df, submit_dataloader):
    t = time.time()
    predicts = []
    for data in tqdm(submit_dataloader):
        user_hash = data['user_id'].to(device)
        item_hash = data['item_id'].to(device)
        city_hash = data['city_id'].to(device)
        item_emb = data["item_emb"].to(device)
        item_emb_seq = data['item_emb_seq'].to(device)
        item_emo_feature = data["item_emo_feature"].to(device)
        sequence_emb = data["sequence_emb"].to(device)
        item2vec = data["item2vec"].to(device)
        predicts.append(F.softmax(
            model(user_hash, item_hash, city_hash, item_emb, item_emo_feature, sequence_emb, item_emb_seq, item2vec)\
                .detach().cpu()).numpy()[:,1].flatten())
    predicts = np.concatenate(predicts)
    ts = (time.time() - t) * 1000.0 / len(predicts) * 2000.0
    print("predict耗时(毫秒)=%s" % ts)
    output_dict = {'Id':submit_df['testSampleId'].astype(str).tolist(), 'result':predicts}
    output_df = pd.DataFrame(output_dict)
    output_path = os.path.join(config.submit_path, "section2-{}.txt".format(config.info))
    output_df.to_csv(output_path, index=False, sep="\t", header=["Id", "result"])
    return output_df, ts

config = get_config()
print(config)

seed = config.seed
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
train_dataloader = get_dataloader(config.rec_data_train_path,config.item_emb_path,config.item_emo_path, config, True)

eval_dataloader = get_dataloader(config.rec_data_eval_path,config.item_emb_path,config.item_emo_path, config, train=True, shuffle=False)
eval_df = pd.read_csv(config.rec_data_eval_path)
submit_df = pd.read_csv(os.path.join(config.submit_path, 'sample.csv'))
submit_dataloader = get_dataloader(os.path.join(config.submit_path, 'sample.csv'),
    config.item_emb_path,config.item_emo_path, config, train=False, shuffle=False)
device = config.device
model = rec_net_emotion_old().to(device)
if config.parallel:
    model = nn.DataParallel(model)
optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=LR, eps=1)
scheduler =  StepLR(optimizer, step_size=20, gamma=0.1)
loss = nn.CrossEntropyLoss(reduction='sum')
max_gAUC = 0
print('start training')
for ep in range(config.epochs):
    model.train()
    print('Epoch {}'.format(ep))
    for data in tqdm(train_dataloader):
        label = data['label'].to(device)
        user_hash = data['user_id'].to(device)
        item_hash = data['item_id'].to(device)
        city_hash = data['city_id'].to(device)
        item_emb = data["item_emb"].to(device)
        item_emb_seq = data['item_emb_seq'].to(device)
        item_emo_feature = data["item_emo_feature"].to(device)
        sequence_emb = data["sequence_emb"].to(device)
        item2vec = data["item2vec"].to(device)
        optimizer.zero_grad()
        pred = model(user_hash, item_hash, city_hash, item_emb, item_emo_feature, sequence_emb, item_emb_seq, item2vec)
        l = loss(pred, label)
        l.backward()
        optimizer.step()
    model.eval()
    gauc = evaluate(model, eval_df, eval_dataloader)
    print('gAUC:', gauc)
    if gauc > max_gAUC:
        max_gAUC = gauc
        predict(submit_df, submit_dataloader)
    scheduler.step()
