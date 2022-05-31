import argparse
import pprint
import yaml


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)

        return Config(**kwargs)


def read_config(path):
    return Config.load(path)


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=20,
                        help='num_epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='learning rate')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='gradient clip norm')
    parser.add_argument('--warm_up', type=float, default=0.01,
                        help='warm up proportion')
    parser.add_argument('--optim', type=str, default='adamw',
                        choices=['adam', 'amsgrad', 'adagrad', 'adamw'],
                        help='optimizer')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--distributed_train', action="store_true")
    parser.add_argument('--seed', type=int, default=2022,
                        help='Random seed')
    parser.add_argument('--model_dir', default='/work/yangshenghao/data/PTM/EmotionRecModel',
                        help='huggingface model name')
    parser.add_argument('--load_ckpt', action="store_true", help='whether to load a trained checkpoint')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--model_path', default='/work/yangshenghao/data/PTM/MaskQueryModel/ckpt/2022-05-07-17:16:40/MQM_epoch_5_3.9207.model', help='name of checkpoint to load')
    parser.add_argument('--print_every', default=200)
    parser.add_argument('--valid_epoch', default=10)
    parser.add_argument('--rec_data_train_path', default='../baseline_recommend/data/train/sample-all.csv')
    parser.add_argument('--rec_data_eval_path', default='../baseline_recommend/data/evaluate/sample-new.csv')
    parser.add_argument('--item_emb_path', default='../baseline_recommend/data/train/item_emb_all.pkl')
    parser.add_argument('--item_emo_path', default='../baseline_recommend/data/train/item_emotion.txt')
    parser.add_argument('--item2vec_path', default='../baseline_recommend/data/train/item_embedding_146790x100_epoch8.pt')
    parser.add_argument('--submit_path', default='../baseline_recommend/data/submit')
    parser.add_argument('--info', default='none')
    parser.add_argument('--parallel', action='store_true', default=False)

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)