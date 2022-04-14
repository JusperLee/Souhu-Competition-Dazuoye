###
# Author: Kai Li
# Date: 2022-04-14 11:19:17
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-04-14 13:46:16
###
import warnings
warnings.filterwarnings("ignore")
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
from datasets import TrainDataset
from model import RobertaATSA
from utils import get_logger, seed_everything, merge_instances, ATSAInstance, AverageMeter
import scipy as sp
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold,StratifiedGroupKFold
from config import CFG
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


def train_fn(fold, train_loader,model, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    # start = end = time.time()
    global_step = 0
    grad_norm = 0
    train_true = []
    train_pred = []
    tk0=tqdm(enumerate(train_loader),total=len(train_loader))
    for step, batch in tk0:
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        batch['label'] = batch['label'].to(device)
        batch['idx'] = batch['idx'].to(device)
            
        batch_size = batch['input_ids'].size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds, loss = model(batch)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
                
        for item in y_preds.detach().cpu().numpy():
            train_pred.append(item.argmax(-1))
        for item in np.array(batch["label"].cpu()):
            train_true.append(item)
        if step % 200 == 0:
            train_pred_tmp = np.array(train_pred)
            train_true_tmp = np.array(train_true)
            avg_acc = accuracy_score(train_true_tmp, train_pred_tmp)
            avg_f1s = f1_score(train_true_tmp, train_pred_tmp, average='macro')
        
        tk0.set_postfix(Epoch=epoch+1, Loss=losses.avg,lr=scheduler.get_lr()[0], ACC=100 * avg_acc, F1=100 * avg_f1s)
    return losses.avg

def valid_fn(valid_loader, model, device):
    losses = AverageMeter()
    model.eval()
    # preds = []
    valid_true = []
    valid_pred = []
    tk0=tqdm(enumerate(valid_loader),total=len(valid_loader))
    for step, batch in tk0:
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        batch['label'] = batch['label'].to(device)
        batch['idx'] = batch['idx'].to(device)
        batch_size = batch['input_ids'].size(0)
        with torch.no_grad():
            y_preds,loss = model(batch)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        batch_pred = y_preds.detach().cpu().numpy()
        for item in batch_pred:
            valid_pred.append(item.argmax(-1))
        for item in np.array(batch[3].cpu()):
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

def train_loop(train_datas, val_datas, fold, LOGGER):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info(f"========== fold: {fold} training ==========")
    LOGGER.info(f"========== loading datasets ==========")
    # ====================================================
    # loader
    # ====================================================
    train_dataset = TrainDataset(train_datas, CFG)
    valid_dataset = TrainDataset(val_datas, CFG)

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
    LOGGER.info(f"========== loadding model ==========")
    best_score = 0.
    model = RobertaATSA(CFG)
    torch.save(CFG, os.path.join("./", CFG.exp_name, "checkpoint")+'/config.pth')
    
    model.to(device)
    
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        parameters=model.named_parameters()
        roberta_parameters=model.roberta.named_parameters()
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in roberta_parameters if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay, 'initial_lr':encoder_lr},
            {'params': [p for n, p in roberta_parameters if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0, 'initial_lr':encoder_lr},
            {'params': [p for n, p in parameters if "roberta" not in n],
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
    
    num_train_steps = int(len(train_dataset) / CFG.batch_size * CFG.epochs)
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
            torch.save(model.state_dict(),os.path.join("./", CFG.exp_name, "checkpoint")+"/model_fold{fold}_best.bin")

    torch.cuda.empty_cache()
    gc.collect()