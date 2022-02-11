#common library
import os
import pprint
import shutil
import copy
import time
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
from common.utility.visualization import show_2d,show_3d,get_joint_prediction_origin
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


#prepare dataset

data_transforms = transforms.Compose([ToTensor(config.dataset.name),                                      
                                      Normalize(config.dataset.name,config.dataiter,config.train)])

dataset_valid = tcDataset(config.dataset,config.dataset.val_image_set,config.network,data_transforms)
valid_data_loader = DataLoader(dataset = dataset_valid, batch_size = config.dataiter.batch_images_per_ctx, shuffle = True,
                                   num_workers = config.dataiter.threads, drop_last = False)

#prepare network
net = get_pose_net(config.network,config.train)
net = DataParallelModel(net).cuda()  # claim multi-gpu in CUDA_VISIBLE_DEVICES
init_pose_net(net, config.network)
net.eval()    
torch.set_grad_enabled(False)

#main demo procedure
MPJPE_val = 0
total_iter_val = 0

#random select the test id to see the result
idx = 5
sample = dataset_valid.__getitem__(idx)
body_center = (sample['GT_label'][7,:] + sample['GT_label'][8,:]) / 2
body_center = torch.unsqueeze(body_center,0)
sample['image'] = torch.unsqueeze(sample['image'],0)
sample['camparam'] = torch.unsqueeze(sample['camparam'],0)

predict_cat,volume_heatmap_cat,vmax_cat,vmin_cat  = net(sample['image'].float().cuda(),sample['camparam'].float().cuda(),body_center.float().cuda())   
predict_vis = get_joint_prediction_origin(predict_cat[len(predict_cat) - 1].cpu(),vmax_cat[:,vmax_cat.shape[1] - 1,:].cpu(),vmin_cat[:,vmin_cat.shape[1] -1,:].cpu())  


#vis the result 
for c in range(4):              
    kcam = sample['camparam'][0][c].cpu().numpy()
    predict_3D = np.vstack((np.transpose(predict_vis[0,:,:].data.numpy()),np.ones((1,config.network.pred_joints))))
    predict_2D = np.dot(kcam,predict_3D) #project 3d prediction to 2d
    predict_2D[0,:] = predict_2D[0,:] / predict_2D[2,:]
    predict_2D[1,:] = predict_2D[1,:] / predict_2D[2,:]
    predict_2D_vis = np.transpose(predict_2D[0:2,:])
    img = sample['image'][0][c].clone()
    img[0,:,:] = ((img[0,:,:] * config.dataiter.std[0]) + config.dataiter.mean[0]) * 255.0
    img[1,:,:] = ((img[1,:,:] * config.dataiter.std[1]) + config.dataiter.mean[1]) * 255.0
    img[2,:,:] = ((img[2,:,:] * config.dataiter.std[2]) + config.dataiter.mean[2]) * 255.0
    img = img.permute(1,2,0).cpu().int().numpy() 
    img = show_2d(img,predict_2D_vis) #show predict 2D
    fig = plt.figure()
    plt.imshow(img)
show_3d(np.transpose(predict_3D[0:3,:])) #show predict 3D

for c in range(4):              
    kcam = sample['camparam'][0][c].cpu().numpy()
    predict_3D = np.vstack((np.transpose(sample['GT_label'].numpy()),np.ones((1,config.network.pred_joints))))
    predict_2D = np.dot(kcam,predict_3D) #project 3d prediction to 2d
    predict_2D[0,:] = predict_2D[0,:] / predict_2D[2,:]
    predict_2D[1,:] = predict_2D[1,:] / predict_2D[2,:]
    predict_2D_vis = np.transpose(predict_2D[0:2,:])
    img = sample['image'][0][c].clone()
    img[0,:,:] = ((img[0,:,:] * config.dataiter.std[0]) + config.dataiter.mean[0]) * 255.0
    img[1,:,:] = ((img[1,:,:] * config.dataiter.std[1]) + config.dataiter.mean[1]) * 255.0
    img[2,:,:] = ((img[2,:,:] * config.dataiter.std[2]) + config.dataiter.mean[2]) * 255.0
    img = img.permute(1,2,0).cpu().int().numpy() 
    img = show_2d(img,predict_2D_vis) #show GT 2D
    fig = plt.figure()
    plt.imshow(img)
show_3d(sample['GT_label'].cpu().numpy()) #show GT 3D
   




        



