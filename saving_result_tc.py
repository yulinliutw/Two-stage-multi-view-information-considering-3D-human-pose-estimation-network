#common library
import os
import pprint
import shutil
import copy
import time
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
from common.utility.visualization import show_2d,show_3d,get_joint_prediction_origin
from spacepy import pycdf
#from common.eval_metrics import mpjpe
from pylab import *
import numpy as np 
from tensorboardX import SummaryWriter

#pytorch
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

#user module
from model.config_pytorch import * #s_config is defined in this lib
from model.common_loss.balanced_parallel import DataParallelModel, DataParallelCriterion
from model.dataloader.tc import tcDataset
from model.dataloader.function_trans import *


#load the parameter setting
config = copy.deepcopy(s_config)
config.network = get_default_network_config(config.train)  # defined in blocks
try:
    s_config_file = './parameter_setting/saving_result.yaml' #set up the config file to load the saving setting
    config = update_config_from_file(config, s_config_file, check_necessity=True)
    config.network.input_width = config.train.patch_width
    config.network.input_height = config.train.patch_height
    print('load config file successfully')
except:
    print('Using default setting')
    
config = update_config_from_args(config, s_args)  # config in argument is superior to config in file

#import dynamic config
exec('from model.main_network.' + config.pytorch.block + \
     ' import get_pose_net, init_pose_net')

#define devices create multi-GPU context
os.environ["CUDA_VISIBLE_DEVICES"] = config.pytorch.gpus  # a safer method
devices = [int(i) for i in config.pytorch.gpus.split(',')]

#prepare network
data_transforms = transforms.Compose([ToTensor(config.dataset.name),                                      
                                      Normalize(config.dataset.name,config.dataiter,config.train)])
net = get_pose_net(config.network,config.train)
net = DataParallelModel(net).cuda()  # claim multi-gpu in CUDA_VISIBLE_DEVICES
init_pose_net(net, config.network)
net.eval()    
torch.set_grad_enabled(False)

#prepare dataset
for a in [1,2,4]:
    Mydataset = tcDataset(config.dataset,config.dataset.val_image_set+'_A'+str(a)+'.csv',config.network,data_transforms)
    My_data_loader = DataLoader(dataset = Mydataset, batch_size = config.dataiter.batch_images_per_ctx, shuffle = False,
                                       num_workers = config.dataiter.threads, drop_last = False)
    predict_vis_saving = np.zeros([config.dataiter.batch_images_per_ctx,config.network.pred_joints*3])
    GT_label_saving = np.zeros([config.dataiter.batch_images_per_ctx,config.network.pred_joints*3])
    for i_batch, sample_batched in enumerate(My_data_loader):    
        body_center = (sample_batched['GT_label'][:,7,:] + sample_batched['GT_label'][:,8,:]) / 2
        predict_cat,volume_heatmap_cat,vmax_cat,vmin_cat = net(sample_batched['image'].float().cuda(),sample_batched['camparam'].float().cuda(),body_center.float().cuda())   
        predict_vis = get_joint_prediction_origin(predict_cat[0].cpu(),vmax_cat[:,0,:].cpu(),vmin_cat[:,0,:].cpu())    
        predict_vis_numpy = predict_vis.view((predict_vis.size()[0],-1)).numpy()    
        GT_label_numpy = sample_batched['GT_label'].view((sample_batched['GT_label'].size()[0],-1)).numpy()
        #saving the predicts and labels to the numpy arrays separately
        if(i_batch == 0):
            predict_vis_saving[:,:] = predict_vis_numpy
            GT_label_saving[:,:] = GT_label_numpy
        predict_vis_saving = np.concatenate((predict_vis_saving,predict_vis_numpy),axis=0)
        GT_label_saving = np.concatenate((GT_label_saving,GT_label_numpy),axis=0)   
        print('Evualation done | iter:'+str(i_batch) + '|action:'+str(a))
    #save the result to the cdf file
    cdf = pycdf.CDF(os.path.join(config.pytorch.output_path,'A'+str(a)+'_pred.cdf'), '')
    cdf['data'] = predict_vis_saving
    cdf.close()
    cdf = pycdf.CDF(os.path.join(config.pytorch.output_path,'A'+str(a)+'_label.cdf'), '')
    cdf['data'] = GT_label_saving
    cdf.close()  