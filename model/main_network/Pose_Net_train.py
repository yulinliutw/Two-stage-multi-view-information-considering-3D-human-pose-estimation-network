import os
from easydict import EasyDict as edict
import time
import torch
import torch.nn as nn
from torchvision.models.resnet import model_urls
import torch.utils.model_zoo as model_zoo

from ..base_modules.color_stream_net_train_multi_feature_train import Feature2D_extracter_model
from ..base_modules.proj_block import proj_splat
from ..base_modules.v2v import V2VModel
from ..base_modules.operator_function import integrate_tensor_3d_with_coordinates

class PoseNet(nn.Module):
    def __init__(self,input_width,input_height,backbone,grid_conv3D_head):
        super(PoseNet, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.backbone = backbone #color stream network         
        self.volume_net = grid_conv3D_head        

    def forward(self, color_imgs,Kcam,body_center):       
        # reshape for backbone forward
        color_imgs = color_imgs.view(-1, *color_imgs.shape[2:])
        
        #get features from each info.
        color_streams  = self.backbone(color_imgs)
        
        #predict the 3D heatmap
        volume_heatmap_cat,vmax_cat,vmin_cat,feature_from_each_stage = self.volume_net(color_streams,Kcam,body_center)   
        
        #predict result in each stage
        output_s1 = integrate_tensor_3d_with_coordinates(volume_heatmap_cat[0])
        output_s2 = integrate_tensor_3d_with_coordinates(volume_heatmap_cat[1])        
        
        #concat the result from each stage       
        output_cat = [output_s1,output_s2]
        
        return output_cat,volume_heatmap_cat,vmax_cat,vmin_cat

def get_pose_net(cfg,cfg_train):
    input_width,input_height = cfg_train.patch_width,cfg_train.patch_height    
    backbone_net = Feature2D_extracter_model(cfg.conv_2D_out_channels)    
    proj_unit = proj_splat(cfg)    
    grid_conv3D_head = V2VModel(proj_unit,cfg.nvox,cfg.conv_2D_out_channels, cfg.pred_joints,cfg.box_range,cfg.volume_aggregation) 
    pose_net = PoseNet(input_width,input_height,backbone_net,grid_conv3D_head)
    return pose_net


def init_pose_net(pose_net, cfg):
    if os.path.exists(cfg.pretrained):
        if(cfg.from_model_zoo == 'openpose'):
            pose_net_dict = pose_net.module.backbone.state_dict()                                 
            model = torch.load(cfg.pretrained)  
            model['model1_2.8.weight'] = model['model1_2.8.weight'][1,:,:,:].view(1,512,1,1) #only take one joint prediction as body center
            model['model1_2.8.bias'] = model['model1_2.8.bias'][1].view(1)            
            #reload weight from pretrain model: 2d feature extractor -> model1_2
            pose_net_dict['model1_2.0.weight'] = model['model0.0.weight']
            pose_net_dict['model1_2.0.bias'] = model['model0.0.bias']
            
            pose_net_dict['model1_2.2.weight'] = model['model0.2.weight']
            pose_net_dict['model1_2.2.bias'] = model['model0.2.bias']
            
            pose_net_dict['model1_2.5.weight'] = model['model0.5.weight']
            pose_net_dict['model1_2.5.bias'] = model['model0.5.bias']
            
            pose_net_dict['model1_2.7.weight'] = model['model0.7.weight']
            pose_net_dict['model1_2.7.bias'] = model['model0.7.bias']
            #reload weight from pretrain model: 2d feature extractor -> model3 
            pose_net_dict['model3.0.weight'] = model['model0.10.weight']
            pose_net_dict['model3.0.bias'] = model['model0.10.bias']
            
            pose_net_dict['model3.2.weight'] = model['model0.12.weight']
            pose_net_dict['model3.2.bias'] = model['model0.12.bias']
            
            pose_net_dict['model3.4.weight'] = model['model0.14.weight']
            pose_net_dict['model3.4.bias'] = model['model0.14.bias']
            
            pose_net_dict['model3.6.weight'] = model['model0.16.weight']
            pose_net_dict['model3.6.bias'] = model['model0.16.bias']
            #reload weight from pretrain model: 2d feature extractor -> model4
            pose_net_dict['model4.1.weight'] = model['model0.19.weight']
            pose_net_dict['model4.1.bias'] = model['model0.19.bias']
            
            pose_net_dict['model4.3.weight'] = model['model0.21.weight']
            pose_net_dict['model4.3.bias'] = model['model0.21.bias']
            
            pose_net_dict['model4.5.weight'] = model['model0.23.weight']
            pose_net_dict['model4.5.bias'] = model['model0.23.bias']
            
            pose_net_dict['model4.7.weight'] = model['model0.25.weight']
            pose_net_dict['model4.7.bias'] = model['model0.25.bias']            
            #update the pretrain weight in current model
            pose_net.module.backbone.load_state_dict(pose_net_dict)             
            print("Init the 2D feature extracter from openpose model", cfg.pretrained)            
        else:
            pose_net_dict = pose_net.state_dict()                                 
            model = torch.load(cfg.pretrained)            
            pretrained_dict = {k: v for k, v in model.items() if k in pose_net_dict}
            pose_net_dict.update(pretrained_dict)
            pose_net.load_state_dict(pose_net_dict)            
            print("Init Network from pretrained", cfg.pretrained)
    else:
        print("Random init the Network")	
