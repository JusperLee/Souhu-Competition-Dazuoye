###
# Author: Kai Li
# Date: 2022-04-14 11:55:59
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-04-14 12:47:30
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

from utils import get_logger, seed_everything, merge_instances, ATSAInstance
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
from trainer import train_loop
import transformers
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
# from  dice_loss import  DiceLoss
transformers.logging.set_verbosity_error()

### 初始化
os.makedirs(os.path.join("./", CFG.exp_name), exist_ok=True)
os.makedirs(os.path.join('./', CFG.exp_name, "checkpoint"), exist_ok=True)
os.makedirs(os.path.join('./', CFG.exp_name, "{}".format(CFG.model.split('/')[-1])), exist_ok=True)
os.environ['TOKENIZERS_PARALLELISM']="true"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed_everything(seed=42)
LOGGER = get_logger(CFG.exp_name+'/train')

Fold = KFold(n_splits=CFG.n_fold)
instances = pickle.load(open('/home/likai/souhu/TMM_for_MAMS/ATSA/train.pickle','rb'))
instances=merge_instances(instances)
train_indexs = {}
val_indexs = {}
for n, (train_index, val_index) in enumerate(Fold.split(instances)):
    train_indexs[n] = train_index
    val_indexs[n] = val_index
instances = np.array(instances)
for fold in range(CFG.n_fold):
    train_loop(instances[train_indexs[fold]].tolist(), instances[val_indexs[fold]].tolist(), fold, LOGGER)
    