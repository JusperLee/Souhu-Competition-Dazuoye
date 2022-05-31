#%%
import numpy as np
import pickle as pk



with open('../baseline_recommend/data/train/item_emb_1536_0528.pkl','rb') as f:
    item_embs_new = pk.load(f)
with open('../baseline_recommend/data/train/item_emb_768_0528.pkl','rb') as f:
    item_embs_old = pk.load(f)
#%%
print(item_embs_new.shape)
print(item_embs_old.shape)
item_embs_all = np.concatenate([item_embs_old, item_embs_new], axis=1)
# %%
print(item_embs_all.shape)
# %%
with open('../baseline_recommend/data/train/item_emb_all_entity.pkl','wb') as f:
    pk.dump(item_embs_all, f)
# %%
import pandas as pd
# train_old = pd.read_csv('../baseline_recommend/data/train/sample.csv')
# train_new = pd.read_csv('../baseline_recommend/data/train/sample-new.csv')
# train_old = train_old.append(train_new)
# train_old.to_csv('../baseline_recommend/data/train/sample-all.csv', index=False)
test_old = pd.read_csv('../baseline_recommend/data/evaluate/sample.csv')
test_new = pd.read_csv('../baseline_recommend/data/evaluate/sample-new.csv')
test_old = test_old.append(test_new)
print(len(test_old))
test_old.to_csv('../baseline_recommend/data/evaluate/sample-all.csv', index=False)
# %%
