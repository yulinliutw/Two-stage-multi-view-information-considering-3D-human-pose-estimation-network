import os
import yaml
import argparse
import copy
import numpy as np
from easydict import EasyDict as edict

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description='model_parameter_set_up')
    # yaml config file
    parser.add_argument('--cfg', help='experiment configure file name', default='./parameter_setting/default.yaml', type=str)
    parser.add_argument('--model', help='path/to/model, requird when resume/test', type=str)
    # parser.add_argument('--autoresume', help='Whether resume from ckpt', default=True, type=str2bool)
    args, rest = parser.parse_known_args()
    return args

# default config
def get_default_network_config(cfg):
    config = edict()
    config.from_model_zoo = ''
    config.pretrained = ''    
    config.conv_2D_out_channels = 31
    config.input_channel = 3
    config.pred_joints = 15
    config.input_width = cfg.patch_width
    config.input_height = cfg.patch_height
    config.box_range = 1600

    config.nvox = 32
    config.volume_aggregation = 'sum'    
    return config

def get_default_config_pytorch():
    config = edict()
    config.gpus = '0,1,2,3'
    config.frequent = 20
    config.output_path = './output/mpii/resnet50v1_ft'
    config.log_path = './log/mpii/resnet50v1_ft'
    config.block = 'Pose_Net'
    config.loss = 'integral'
    return config

def get_default_dataset_config():
    config = edict()
    config.name = 'human 3.6m'
    config.root_dir = ''
    config.cam_num = [1,2,3,4]    
    config.train_image_set = ''    
    config.test_image_set = ''
    config.val_image_set = ''
    config.cam_matrix_path = ''
    return config

def get_default_dataiter_config():
    config = edict()
    config.epoch = 50
    config.batch_images_per_ctx = 32
    config.threads = 4
    config.use_color_normalize = True
    config.mean = np.array([0.485 , 0.456, 0.406])
    config.std = np.array([0.229 , 0.224 , 0.225])
    return config

def get_default_optimizer_config():
    config = edict()
    config.lr = 0.001
    # change learning rate when training of the nth epoch is finished.
    config.lr_epoch_step = ''
    config.lr_factor = 0.1

    config.optimizer_name = 'sgd'
    config.momentum = 0.9
    config.wd = 0.0001
    config.gamma1 = 0.99
    config.gamma2 = 0.0
    return config

def get_default_train_config():
    config = edict()
    config.record_freq = 200
    # start with the begin_epoch-th epoch. Init with the (begin_epoch - 1)-th epoch if begin_epoch >= 2.
    config.begin_epoch = 1
    # end when finishing the end_epoch-th epoch.
    config.end_epoch = 1

    config.model_prefix = 'model'
    config.resume = False

    # config.image_res = np.array([256, 256])  # width * height, ex: 192 * 256
    config.patch_width = 256
    config.patch_height = 256

    return config

def get_default_test_config():
    config = edict()
    config.feat_out = {}
    return config

def update_config_from_file(_config, config_file, check_necessity=True):
    config = copy.deepcopy(_config)
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        if vk in config[k]:
                            if isinstance(vv, list) and not isinstance(vv[0], str):
                                config[k][vk] = np.asarray(vv)
                            else:
                                config[k][vk] = vv
                        else:
                            if check_necessity:
                                raise ValueError("{}.{} not exist in config".format(k, vk))
                else:
                    raise ValueError("{} is not dict type".format(v))
            else:
                if check_necessity:
                    raise ValueError("{} not exist in config".format(k))
    return config

def update_config_from_args(_config, args):
    config = copy.deepcopy(_config)
    return config

# 1. parsing arguments
s_args = parse_args()
s_config_file = s_args.cfg
# 2. parsing pytorch config
s_config = edict()
s_config.pytorch = get_default_config_pytorch()
s_config.dataset = get_default_dataset_config()
s_config.dataiter = get_default_dataiter_config()
s_config.optimizer = get_default_optimizer_config()
s_config.train = get_default_train_config()
s_config.test = get_default_test_config()
try:
    s_config = update_config_from_file(s_config, s_config_file, check_necessity=False)    
except:
    print('load cfg file error!!')



def get_base_common_config(config_file):
    base_config = edict()
    base_config.pytorch = get_default_config_pytorch()
    base_config.dataset = get_default_dataset_config()
    base_config.dataiter = get_default_dataiter_config()
    base_config.optimizer = get_default_optimizer_config()
    base_config.train = get_default_train_config()
    base_config.test = get_default_test_config()    
    base_config = update_config_from_file(base_config, config_file, check_necessity=False)
    return base_config
