###
# Author: Kai Li
# Date: 2022-04-14 11:23:43
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-04-14 15:03:55
###
#=================参数设置=================
class CFG:
    exp_name="chinese-roberta-wwm-extt-test"
    apex=True
    num_workers=0
    model="hfl/chinese-roberta-wwm-ext"    # huggingface 预训练模型,可以访问 https://huggingface.co/models 选择更多其他模型
    scheduler='linear'                   # ['linear', 'cosine'] # lr scheduler 类型
    batch_scheduler=True                 # 是否每个step结束后更新 lr scheduler
    num_cycles=0.5                       # 如果使用 cosine lr scheduler， 该参数决定学习率曲线的形状，0.5代表半个cosine曲线
    num_warmup_steps=0                   # 模型刚开始训练时，学习率从0到初始最大值的步数
    epochs=5 
    last_epoch=-1                        # 从第 last_epoch +1 个epoch开始训练
    encoder_lr=2e-5                      # 预训练模型内部参数的学习率
    decoder_lr=2e-5                      # 自定义输出层的学习率
    batch_size=48                       
    max_len=512                     
    weight_decay=0.01        
    gradient_accumulation_steps=1        # 梯度累计步数，1代表每个batch更新一次
    # max_grad_norm=1000  
    seed=42 
    n_fold=4                             # 总共划分数据的份数
    trn_fold=[0,1,2,3]                   # 需要训练的折数，比如一共划分了4份，则可以对应训练4个模型，1代表用编号为1的折做验证，其余折做训练
    train=True
    # model
    ROBERTA_DIM=768
    LINEAR_HIDDEN_DIM=256