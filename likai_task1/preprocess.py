###
# Author: Kai Li
# Date: 2022-04-14 13:50:45
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-04-14 13:50:46
###
import gc
import pickle
import json
from tqdm import tqdm
import pandas as pd
from configs import BaseConfigs
from instance import ACSAInstance, ATSAInstance
import xml.etree.ElementTree

class Preprocessor(object):
    def __init__(self,configs):
        self.configs=configs

    def preprocess_ACSA(self,src,dst):
        instances=[]
        with open(src,'r',encoding='utf8') as f:
            tree=xml.etree.ElementTree.parse(f)
            for node in tqdm(tree.iterfind('sentence')):
                sentence=node.findtext('text')

                for category_node in node.iterfind('aspectCategories/aspectCategory'):
                    instance = ACSAInstance()
                    instance.sentence=sentence
                    instance.category=ACSAInstance.category2index[category_node.get('category')]
                    instance.polarity=ACSAInstance.polarity2index[category_node.get('polarity')]

                    instances.append(instance)

        with open(dst,'wb') as f:
            pickle.dump(instances,f)

    def preprocess_ATSA(self,src,dst):
        instances=[]
        data = pd.read_csv(src, sep=r'\n', names=['all'])
        data['all'] = data['all'].apply(lambda x: json.loads(x))
        data['content'] = data['all'].apply(lambda x: x['content'])
        data['entities'] = data['all'].apply(lambda x: [(key, label) for key, label in dict(x['entity']).items()])
        data = data.explode('entities').reset_index(drop=True)
        data['entity'] = data['entities'].str[0]
        data['label'] = data['entities'].str[1] + 2
        del data['all'], data['entities']; gc.collect()
        for idx in tqdm(range(len(data['content']))):
            instance = ATSAInstance()
            # instance.sentence=data['content'][idx]
            instance.aspect_term=data['entity'][idx]
            instance.aspect_from_pos=data['content'][idx].find(data['entity'][idx])
            instance.polarity=ATSAInstance.polarity2index[data['label'][idx]]
            if len(data['content'][idx]) > 508:
                start = data['content'][idx].find(data['entity'][idx])
                before = data['content'][idx][:start+len(data['entity'][idx])+1]
                after = data['content'][idx][start+len(data['entity'][idx])+1:]
                if len(before) > 254:
                    before = before[-1:-255:-1]
                if len(after) > 254:
                    after = after[:254]
                instance.sentence=before+after
                # import pdb; pdb.set_trace()
            else:
                instance.sentence=data['content'][idx]
            
            instances.append(instance)

        with open(dst,'wb') as f:
            pickle.dump(instances,f)

if __name__=='__main__':
    configs=BaseConfigs()
    processor=Preprocessor(configs)
    processor.preprocess_ATSA('/home/likai/souhu/emo/Sohu2022_data/nlp_data/train.txt','ATSA/train.pickle')
