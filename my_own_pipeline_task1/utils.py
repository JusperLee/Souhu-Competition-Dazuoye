###
# Author: Kai Li
# Date: 2022-04-14 11:24:35
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-04-14 12:35:09
###
import warnings
warnings.filterwarnings('ignore')
import json
import os
import torch
import numpy as np
import random

class ATSAInstance(object):
    polarity2index={0:0,1:1,2:2,3:3,4:4,5:5}
    sentence=None
    word_tokens = None
    bert_tokens = None
    aspect_term=None
    aspect_from=None
    aspect_to=None
    aspect_from_pos=None
    aspect_bert_tokens=None
    polarity=None

#======生成log文件记录训练输出======
def get_logger(filename):
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

#=======设置全局seed保证结果可复现====
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def merge_instances(instances):
    instances_by_sentence = []
    for x in instances:
        instances_by_sentence.append(x)
    return instances_by_sentence

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