import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import pickle as pk
from torch.utils.data import Dataset, DataLoader
from utils import *
from model import *
from params import *
from tqdm import tqdm
import tensorflow as tf
if tf.__version__ >= '2.0':
  tf = tf.compat.v1
from tensorflow import feature_column as fc


data_train = pd.read_csv('./data/train/sample.csv')
data_eval = pd.read_csv('./data/evaluate/sample.csv')

train_label = data_train['label'].to_numpy()
eval_label = data_eval['label'].to_numpy()
with open('./data/train/features.pk', 'rb') as f:
    fc_train = pk.load(f)
with open('./data/evaluate/features.pk', 'rb') as f:
    fc_eval = pk.load(f)



def evaluate(model, df, eval_dataloader):
    input_path = TEST_FILE
    df = pd.read_csv(input_path, delimiter=",", usecols=["pvId", "label"])
    pvid_list = df['pvId'].astype(str).tolist()
    predicts = []
    for x, _ in tqdm(eval_dataloader):
        x = x.to(device)
        predicts.append(model(x).detach().cpu().numpy()[:,1])
    predicts = np.concatenate(predicts)
    labels = df["label"].values
    gauc = evaluation.gAUC(labels, predicts, pvid_list)
    return gauc


def predict(self):
    submit_dir = './data/submit'
    input_path = os.path.join(submit_dir, "sample.csv")
    id_df = pd.read_csv(input_path, delimiter=",", usecols=["testSampleId"])
    t = time.time()
    predicts = self.estimator.predict(
        input_fn=lambda: self.input_fn_predict(input_path)
    )
    predicts_df = pd.DataFrame.from_dict(predicts)
    logits = predicts_df["logistic"].map(lambda x: x[0])
    # 计算2000条样本平均预测耗时（毫秒）
    ts = (time.time() - t) * 1000.0 / len(id_df) * 2000.0
    output_df = pd.concat([id_df, logits], axis=1)
    output_path = os.path.join(submit_dir, "section2.txt")
    output_df.to_csv(output_path, index=False, sep="\t", header=["Id", "result"])
    print("predict耗时(毫秒)=%s" % ts)
    return output_df, ts


ds_train = file_to_dataset(TRAIN_FILE, isTest=False, shuffle=False, batch_size=FLAGS.batch_size * 10,
                                    num_epochs=1)

ds_test = file_to_dataset(TEST_FILE, isTest=False, shuffle=False, batch_size=FLAGS.batch_size * 10, num_epochs=1)

train_dataset = sougou_dataset(fc_train, train_label)
eval_dataset = sougou_dataset(fc_eval, np.zeros(fc_eval.shape[0]))

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SZ, shuffle=True, drop_last=False)
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SZ * 10, shuffle=False, drop_last=False)

model = rec_net(len(fc_train[0])).to(device)
optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=LR, eps=1)
# loss = nn.BCELoss(reduction='sum')
loss = nn.CrossEntropyLoss(reduction='sum')
print('start training')
MAX_ITER = 1e10
for ep in range(EPOCH):
    model.train()
    it = 0
    for x, label in tqdm(train_dataloader):
        optimizer.zero_grad()
        label = label.to(device)
        x = x.to(device)
        pred = model(x)
        l = loss(pred, label)
        print('loss:', l)
        l.backward()
        optimizer.step()
        if it > MAX_ITER:
            break
        it += 1
        # if it % 100 == 0:
        #     print('loss:', l)
    model.eval()
    gauc = evaluate(model, data_eval, eval_dataloader)
    print('gAUC:', gauc)


