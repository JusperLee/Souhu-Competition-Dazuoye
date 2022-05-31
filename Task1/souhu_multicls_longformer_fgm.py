# %%
import os

OUTPUT_DIR = './souhu_multicls_longformer_focal_fgm/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# %%
import pandas as pd
import numpy as np
import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold,StratifiedGroupKFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
import transformers
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
# from  dice_loss import  DiceLoss
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
#=================参数设置=================
class CFG:
    apex=True
    num_workers=8
    model="longformer-chinese-base-4096"    # huggingface 预训练模型
    scheduler='cosine'                   # ['linear', 'cosine'] # lr scheduler 类型
    batch_scheduler=True                 # 是否每个step结束后更新 lr scheduler
    num_cycles=0.5                       # 如果使用 cosine lr scheduler， 该参数决定学习率曲线的形状，0.5代表半个cosine曲线
    num_warmup_steps=0                   # 模型刚开始训练时，学习率从0到初始最大值的步数
    epochs=5 
    last_epoch=-1                        # 从第 last_epoch +1 个epoch开始训练
    encoder_lr=2e-5                      # 预训练模型内部参数的学习率
    decoder_lr=2e-5                      # 自定义输出层的学习率
    batch_size=32                       
    max_len=600                     
    weight_decay=0.01        
    gradient_accumulation_steps=1        # 梯度累计步数，1代表每个batch更新一次
    max_grad_norm=5  
    seed=42 
    n_fold=3                             # 总共划分数据的份数
    trn_fold=[0,1,2,3]                   # 需要训练的折数，比如一共划分了4份，则可以对应训练4个模型，1代表用编号为1的折做验证，其余折做训练
    train=False

# %%
import json
import warnings

warnings.filterwarnings('ignore')
#======生成log文件记录训练输出======
def get_logger(filename=OUTPUT_DIR+'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger()
#=======设置全局seed保证结果可复现====
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)
#=====将官方txt数据转换成我们所需的格式==
def get_train_data(input_file):
    corpus = []
    labels = []
    entitys = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            tmp = json.loads(line.strip())
            raw_contents = tmp['content']
            raw_entitys = tmp['entity']
            label = int(tmp["label"])
            if label == -2:
                label = 4
            elif label == -1:
                label = 3
            for entity in [raw_entitys]:
                text = raw_contents.strip()
                corpus.append(text)
                entitys.append(entity)
                labels.append(label)
    assert len(corpus) == len(labels) == len(entitys)
    return corpus, labels, entitys


def get_test_data(input_file):
    ids = []
    corpus = []
    entitys = []
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tmp = json.loads(line.strip())
            raw_id = tmp['id']
            raw_contents = tmp['content']
            raw_entitys = tmp['entity']
            for entity in [raw_entitys]:
                text = raw_contents.strip()
                corpus.append(text)
                ids.append(raw_id)
                entitys.append(entity)
    assert len(corpus) == len(entitys) == len(ids)
    return corpus, entitys, ids

# %%
# 读取txt文件并处理成 文本-情感关键词-情感类别 一一对应的数据
train_corpus, train_labels, train_entitys = get_train_data(input_file='./data/generated_train_data.txt')
test_corpus, test_entitys, test_ids = get_test_data(input_file='./data/generated_test_data.txt')

train = {'content':train_corpus,'entity':train_entitys,'label':train_labels}
train = pd.DataFrame(train)

test = {'id':test_ids,'content':test_corpus,'entity':test_entitys}
test = pd.DataFrame(test)

# %%
# 使用gkf，保证同一条文本不出现在不同fold，并且每个fold里5种情感类别标签比例一致
Fold = GroupKFold(n_splits=CFG.n_fold)
groups = train['content'].values
for n, (train_index, val_index) in enumerate(Fold.split(train, train['label'], groups)):
    train.loc[val_index, 'fold'] = int(n)
train['fold'] = train['fold'].astype(int)

# %%
# 载入预训练模型的分词器
tokenizer = AutoTokenizer.from_pretrained(CFG.model)
CFG.tokenizer = tokenizer

# %%
# 使用HF tokenzier 对输入（ 文本+ 情感关键词）进行编码，同一处理成CFG里定义的最大长度
def prepare_input(cfg, text, feature_text):
    inputs = cfg.tokenizer(text, feature_text, 
                           add_special_tokens=True,
                           truncation = True,
                           max_length=CFG.max_len,
                           padding="max_length",
                           return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs
# 将tokenizer编码完成的输入 以及 对应的情感标签 处理成tensor，供模型训练
class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.entitys = df['entity'].values
        self.contents = df['content'].values
        self.labels = df['label'].values

    def __len__(self):
        return len(self.entitys)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, 
                               self.contents[item], 
                               self.entitys[item])
        labels = torch.tensor(self.labels[item], dtype=torch.long)
        return inputs, labels

# %%
# 定义模型结构，该结构是取预训练模型最后一层encoder输出，形状为[batch_size, sequence_length, hidden_size]，
# 在1维取平均，得到[batch_size, hidden_size]的特征向量，传递给分类层得到[batch_size, 5]的向量输出，代表每条文本在五个类别上的得分，最后使用softmax将得分规范化
# 训练过程中额外对 取平均后的输出做了5次dropout，并计算五次loss取平均，该方法可以加速模型收敛，相关思路可参考论文： https://arxiv.org/pdf/1905.09788.pdf

class FGM:
    def __init__(self, model: nn.Module, eps=1.):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.backup = {}

    # only attack word embedding
    def attack(self, emb_name='pretrain_model'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='pretrain_model'):
        for name, para in self.model.named_parameters():
            if para.requires_grad and emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]

        self.backup = {}

class AttentionHead(nn.Module):
    def __init__(self, h_size, hidden_dim=512):
        super().__init__()
        self.W = nn.Linear(h_size, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        
    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector



class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(Focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        self.fc = nn.Linear(self.config.hidden_size*8, 5)
        self._init_weights(self.fc)
        self.drop1=nn.Dropout(0.1)
        self.drop2=nn.Dropout(0.2)
        self.drop3=nn.Dropout(0.3)
        self.drop4=nn.Dropout(0.4)
        self.drop5=nn.Dropout(0.5)
        self.head = AttentionHead(self.config.hidden_size*4, self.config.hidden_size)
        self.focal_loss = Focal_loss(num_classes=5)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        all_hidden_states = torch.stack(outputs.hidden_states)
        cat_over_last_layers = torch.cat(
            (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]),-1
        )
        
        cls_pooling = cat_over_last_layers[:, 0]   
        head_logits = self.head(cat_over_last_layers)
        feature = torch.cat([head_logits, cls_pooling], -1)
        return feature
    
    def loss(self,logits,labels):
        loss_fnc = nn.CrossEntropyLoss()
        loss_fc = self.focal_loss(logits, labels)
        loss = 0.9 * loss_fnc(logits, labels) + 0.1 * loss_fc
        return loss

    def forward(self, inputs,labels=None):
        feature = self.feature(inputs)
        logits1 = self.fc(self.drop1(feature))
        logits2 = self.fc(self.drop2(feature))
        logits3 = self.fc(self.drop3(feature))
        logits4 = self.fc(self.drop4(feature))
        logits5 = self.fc(self.drop5(feature))
        output = self.fc(feature)
        output = F.softmax(output, dim=1)
        _loss=0
        if labels is not None:
            loss1 = self.loss(logits1,labels)
            loss2 = self.loss(logits2,labels)
            loss3 = self.loss(logits3,labels)
            loss4 = self.loss(logits4,labels)
            loss5 = self.loss(logits5,labels)
            _loss = (loss1 + loss2 + loss3 + loss4 + loss5)/5
            
        return output

# %%
# 模型训练常用工具类，记录指标变化
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# %%
def train_fn(fold, train_loader,model, optimizer, epoch, scheduler, device):
    fgm = FGM(model=model)
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    # start = end = time.time()
    global_step = 0
    grad_norm = 0
    tk0=tqdm(enumerate(train_loader),total=len(train_loader))
    for step, (inputs, labels) in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds,loss = model(inputs,labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        fgm.attack()
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds, loss_adv =  model(inputs,labels)
        scaler.scale(loss_adv).backward()
        fgm.restore()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        if step % 50 == 0:
            avg_acc = accuracy_score(labels.cpu().detach().numpy().reshape(-1), y_preds.cpu().detach().numpy().argmax(-1))
            avg_f1s = f1_score(labels.cpu().detach().numpy(), y_preds.cpu().detach().numpy().argmax(-1), average='macro')
        tk0.set_postfix(Epoch=epoch+1, Loss=losses.avg,lr=scheduler.get_lr()[0], ACC=avg_acc, F1=avg_f1s)
    return losses.avg

def valid_fn(valid_loader, model, device):
    losses = AverageMeter()
    model.eval()
    # preds = []
    valid_true = []
    valid_pred = []
    tk0=tqdm(enumerate(valid_loader),total=len(valid_loader))
    for step, (inputs, labels) in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds,loss = model(inputs,labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        batch_pred = y_preds.detach().cpu().numpy()
        for item in batch_pred:
            valid_pred.append(item.argmax(-1))
        for item in np.array(labels.cpu()):
            valid_true.append(item)
        tk0.set_postfix(Loss=losses.avg)
    print('Test set: Average loss: {:.4f}'.format(losses.avg))
    valid_true = np.array(valid_true)
    valid_pred = np.array(valid_pred)
    avg_acc = accuracy_score(valid_true, valid_pred)
    avg_f1s = f1_score(valid_true, valid_pred, average='macro')

    print('Average: Accuracy: {:.3f}%, F1Score: {:.3f}'.format(100 * avg_acc, 100 * avg_f1s))
    print(classification_report(valid_true, valid_pred))

    return avg_acc, avg_f1s, losses.avg


# %%
def train_loop(folds, fold):
    
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    
    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    print(len(train_loader),len(valid_loader))

    # ====================================================
    # model & optimizer
    # ====================================================
    best_score = 0.
    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, OUTPUT_DIR+'config.pth')
    
    model.to(device)
    
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay, 'initial_lr':encoder_lr},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0, 'initial_lr':encoder_lr},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0, 'initial_lr':decoder_lr}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr, 
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters)
    
    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler=='linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        else :
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles, last_epoch = ((cfg.last_epoch+1)/cfg.epochs)*num_train_steps
            )
        return scheduler
    
    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================

    for epoch in range(CFG.epochs-1-CFG.last_epoch):

        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model, optimizer, epoch, scheduler, device)


        # eval
        avg_acc, avg_f1s, valid_loss = valid_fn(valid_loader, model, device)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f} time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {avg_f1s:.4f}')
        
        if best_score < avg_f1s:
            best_score = avg_f1s
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: f1: {best_score:.4f} Model')
            torch.save(model.state_dict(),OUTPUT_DIR+f"model_fold{fold}_best.bin")

    torch.cuda.empty_cache()
    gc.collect()

# %%
if __name__ == '__main__':
    if CFG.train:
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                train_loop(train, fold)

# %%
class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.entitys = df['entity'].values
        self.contents = df['content'].values

    def __len__(self):
        return len(self.entitys)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, 
                               self.contents[item], 
                               self.entitys[item])
        return inputs
def test_and_save_reault(device, test_loader, test_ids, result_path):
    raw_preds = []
    test_pred = []
    for fold in range(CFG.n_fold):
        current_idx = 0
        
        model = CustomModel(CFG, config_path=OUTPUT_DIR+'config.pth', pretrained=True)
        model.to('cuda')
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"model_fold{fold}_best.bin"),map_location=torch.device('cuda')))
        model.eval()
        tk0 = tqdm(test_loader, total=len(test_loader))
        for inputs in tk0:
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            with torch.no_grad():
                y_pred_pa_all = torch.nn.parallel.data_parallel(model, inputs, device_ids=[0,1])
            batch_pred = (y_pred_pa_all.detach().cpu().numpy())/CFG.n_fold
            if fold == 0:
                raw_preds.append(batch_pred)
            else:
                raw_preds[current_idx] += batch_pred
                current_idx += 1
    for preds in raw_preds:
        for item in preds:
            test_pred.append(item.argmax(-1))
    assert len(test_entitys) == len(test_pred) == len(test_ids)
    result = {}
    for id, entity, pre_lable in zip(test_ids, test_entitys, test_pred):
        if pre_lable == 3:
            pre_lable = int(-1)
        elif pre_lable == 4:
            pre_lable = int(-2)
        else:
            pre_lable = int(pre_lable)
        if id in result.keys():
            result[id][entity] = pre_lable
        else:
            result[id] = {entity: pre_lable}
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("id	result")
        f.write('\n')
        for k, v in result.items():
            f.write(str(k) + '	' + json.dumps(v, ensure_ascii=False) + '\n')
    print(f"保存文件到:{result_path}")

# %%
# valid
test_dataset = TestDataset(CFG, test)
test_loader = DataLoader(test_dataset,
                  batch_size=680,
                  shuffle=False,
                  num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
test_and_save_reault(device, test_loader, test_ids, OUTPUT_DIR+'output.txt')
print("+++ bert valid done +++")
