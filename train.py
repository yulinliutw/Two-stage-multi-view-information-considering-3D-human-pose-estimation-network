#common library#
import os
import pprint
import shutil
import copy
import time
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
from common.utility.visualization import show_2d,show_3d,get_joint_prediction_origin
from common.eval_metrics import mpjpe
from pylab import *
import numpy as np 
from tensorboardX import SummaryWriter

#pytorch#
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

#user module#
from model.config_pytorch import * #s_config is defined in this lib
from model.common_loss.balanced_parallel import DataParallelModel, DataParallelCriterion
from model.loss.loss_lib import *
from model.dataloader.h36 import h36Dataset
from model.dataloader.tc import tcDataset
from model.dataloader.function_trans import *
from model.optimizer import get_optimizer



#load the parameter setting#
config = copy.deepcopy(s_config)
config.network = get_default_network_config(config.train)  # defined in blocks
try:
    config = update_config_from_file(config, s_config_file, check_necessity=True)
    config.network.input_width = config.train.patch_width
    config.network.input_height = config.train.patch_height
    print('load config file successfully')
except:
    print('Using default setting')
config = update_config_from_args(config, s_args)  # config in argument is superior to config in file

#import dynamic config#
exec('from model.main_network.' + config.pytorch.block + \
     ' import get_pose_net, init_pose_net')

#define devices create multi-GPU context#
os.environ["CUDA_VISIBLE_DEVICES"] = config.pytorch.gpus  # a safer method
devices = [int(i) for i in config.pytorch.gpus.split(',')]

#prepare tensorboard visualization#
#command: tensorboard --logdir= path#
writer = SummaryWriter(config.pytorch.log_path)

#prepare dataset#

data_transforms = transforms.Compose([ToTensor(config.dataset.name),                                      
                                      Normalize(config.dataset.name,config.dataiter,config.train)])

dataset_train = tcDataset(config.dataset,config.dataset.train_image_set,config.network,data_transforms)
dataset_valid = tcDataset(config.dataset,config.dataset.val_image_set,config.network,data_transforms)
train_data_loader = DataLoader(dataset = dataset_train, batch_size = config.dataiter.batch_images_per_ctx, shuffle = True,
                                   num_workers = config.dataiter.threads, drop_last = True)
valid_data_loader = DataLoader(dataset = dataset_valid, batch_size = config.dataiter.batch_images_per_ctx, shuffle = False,
                                   num_workers = config.dataiter.threads, drop_last = False)

#prepare network#

net = get_pose_net(config.network,config.train)
net = DataParallelModel(net).cuda()  # claim multi-gpu in CUDA_VISIBLE_DEVICES
init_pose_net(net, config.network)

#define optimizer#
optimizer, scheduler = get_optimizer(config.optimizer, net)
vce_loss_multi_stage = VolumetricCELoss_multi_stage()
l1_loss_multi_stage = L1_loss_multi_stage()
ban_loss_multi_stage = bones_angle_loss_multi_stage()
blen_loss_multi_stage = bone_length_loss_multi_stage(config.network)

#main training procedure#
Total_iter = 0
endt1 = 0
endt2 = 0
load_data_T = 0
training_time = 0

print("train epoch:"+str(config.dataiter.epoch))
for ep in range(config.dataiter.epoch):
    beginT_ep = time.time()    
    net.train()    
    torch.set_grad_enabled(True)
    for i_batch, sample_batched in enumerate(train_data_loader):        
        Total_iter = Total_iter + 1
        load_data_end = time.time() - load_data_T
        beginT = time.time()
        #optimize procedure
        body_center = (sample_batched['GT_label'][:,7,:] + sample_batched['GT_label'][:,8,:]) / 2
        predict_cat,volume_heatmap_cat,vmax_cat,vmin_cat = net(sample_batched['image'].float().cuda(),sample_batched['camparam'].float().cuda(),body_center.float().cuda())        
        main_loss = l1_loss_multi_stage(predict_cat,sample_batched['GT_label'].float().cuda(),vmax_cat,vmin_cat) 
        heatmap_vol_loss = vce_loss_multi_stage(volume_heatmap_cat,sample_batched['GT_label'].float().cuda(),vmax_cat,vmin_cat)        
        geo_1_loss = ban_loss_multi_stage (predict_cat,sample_batched['GT_label'].float().cuda(),vmax_cat,vmin_cat)
        geo_2_loss = blen_loss_multi_stage (predict_cat,sample_batched['GT_label'].float().cuda(),vmax_cat,vmin_cat)
        loss = main_loss + heatmap_vol_loss + geo_1_loss + geo_2_loss
                
        optimizer.zero_grad()        
        loss.backward()       
        optimizer.step()
        predict_vis = get_joint_prediction_origin(predict_cat[0].cpu(),vmax_cat[:,0,:].cpu(),vmin_cat[:,0,:].cpu())  
        MPJPE_1 = mpjpe(predict_vis,sample_batched['GT_label'].float())
        predict_vis = get_joint_prediction_origin(predict_cat[len(predict_cat) - 1].cpu(),vmax_cat[:,vmax_cat.shape[1] - 1,:].cpu(),vmin_cat[:,vmin_cat.shape[1] -1,:].cpu())  
        MPJPE_2 = mpjpe(predict_vis,sample_batched['GT_label'].float()) 
        #saving the result
        if((i_batch+1)% (config.train.record_freq) == 0):
            #record the loss value
            writer.add_scalar('Training set/l1_loss', main_loss, Total_iter) 
            writer.add_scalar('Training set/heatmap_vol_loss', heatmap_vol_loss, Total_iter)            
            writer.add_scalar('Training set/geo_1_loss_bones_angle', geo_1_loss, Total_iter) 
            writer.add_scalar('Training set/geo_2_loss_bone_length', geo_2_loss, Total_iter)             
            writer.add_scalar('Training set/loss_predict and label', loss, Total_iter)            
            #saving the pretrain weight
            filename = 'checkpoint_ep'+str(ep)+'_itir_'+str(i_batch)+'.pkl'            
            filename = os.path.join(config.pytorch.output_path, filename)  
            torch.save(net.state_dict(), filename)              
        endt1 = time.time() - beginT               
        print("epoch:"+str(ep)+"||step:"+str(i_batch)+"||loss:"+str(loss.cpu().data.numpy())+"||MPJPE_1:" + str(MPJPE_1.cpu().data.numpy()) +"mm"+"||MPJPE_2:" + str(MPJPE_2.cpu().data.numpy()) +"mm"
              "||training time:"+str(endt1)+"s") 
        del loss
        torch.cuda.empty_cache()
    scheduler.step()        
    
    net.eval()    
    torch.set_grad_enabled(False)
    total_iter_val = 0
    loss_val = 0
    MPJPE_val_1 = 0
    MPJPE_val_2 = 0
    beginT = time.time()
    for i_batch, sample_batched in enumerate(valid_data_loader):
        print("eval:||iter:"+str(i_batch))
        body_center = (sample_batched['GT_label'][:,7,:] + sample_batched['GT_label'][:,8,:]) / 2
        predict_cat,volume_heatmap_cat,vmax_cat,vmin_cat  = net(sample_batched['image'].float().cuda(),sample_batched['camparam'].float().cuda(),body_center.float().cuda())
        
        main_loss = l1_loss_multi_stage(predict_cat,sample_batched['GT_label'].float().cuda(),vmax_cat,vmin_cat) 
        heatmap_vol_loss = vce_loss_multi_stage(volume_heatmap_cat,sample_batched['GT_label'].float().cuda(),vmax_cat,vmin_cat)        
        geo_1_loss = ban_loss_multi_stage (predict_cat,sample_batched['GT_label'].float().cuda(),vmax_cat,vmin_cat)
        geo_2_loss = blen_loss_multi_stage (predict_cat,sample_batched['GT_label'].float().cuda(),vmax_cat,vmin_cat)
        loss = main_loss + heatmap_vol_loss + geo_1_loss + geo_2_loss
        
        predict_vis = get_joint_prediction_origin(predict_cat[0].cpu(),vmax_cat[:,0,:].cpu(),vmin_cat[:,0,:].cpu()) 
        MPJPE_val_1 = MPJPE_val_1 + mpjpe(predict_vis,sample_batched['GT_label'].float()) 
        predict_vis = get_joint_prediction_origin(predict_cat[len(predict_cat) - 1].cpu(),vmax_cat[:,vmax_cat.shape[1] - 1,:].cpu(),vmin_cat[:,vmin_cat.shape[1] -1,:].cpu()) 
        MPJPE_val_2 = MPJPE_val_2 + mpjpe(predict_vis,sample_batched['GT_label'].float())        
        loss_val = loss_val + loss
        total_iter_val = total_iter_val +1       
        
    endt2 = time.time() - beginT    
    loss_val = loss_val / total_iter_val
    MPJPE_val_1 = MPJPE_val_1 / total_iter_val
    MPJPE_val_2 = MPJPE_val_2 / total_iter_val
    writer.add_scalar('Val set/loss_predict and label', loss_val, ep)
    writer.add_scalar('Val set/MPJPE_s1', MPJPE_val_1, ep)  
    writer.add_scalar('Val set/MPJPE_s2', MPJPE_val_2, ep) 
    del loss_val
    torch.cuda.empty_cache()
    training_time = training_time + ((time.time() - beginT_ep)/3600)
writer.close()
print("total training time: " + str(training_time) + "hr")

