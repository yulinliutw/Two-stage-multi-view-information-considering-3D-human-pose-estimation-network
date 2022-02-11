import os
from easydict import EasyDict as edict
import time
import torch
import torch.nn as nn
from torchvision.models.resnet import model_urls
import torch.utils.model_zoo as model_zoo

from ..base_modules.color_stream_net import Feature2D_extracter_model
from ..base_modules.proj_block_org import proj_splat
from ..base_modules.grid_conv3D_head import grid_conv3D
from ..base_modules.v2v import V2VModel
from ..base_modules.operator_function import integrate_tensor_3d_with_coordinates,volume_aggregation_method


class PoseNet(nn.Module):
    def __init__(self,input_width,input_height,backbone,proj_unit,volume_aggregation_cfg,grid_conv3D_head):
        super(PoseNet, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.backbone = backbone #color stream network 
        self.proj_unit = proj_unit
        self.volume_net = grid_conv3D_head
        self.volume_aggregation_cfg = volume_aggregation_cfg

    def forward(self, color_imgs,Kcam):
        batch_size, n_views = color_imgs.shape[:2]
        # reshape for backbone forward
        color_imgs = color_imgs.view(-1, *color_imgs.shape[2:])
        Kcam = Kcam.view(-1,*Kcam.shape[2:])
        #get features from each info.
        body_center,color_streams = self.backbone(color_imgs)        
        
        #project the 2d features to form a 3d feature        
        project3D_grids = self.proj_unit(color_streams,Kcam)               
        # reshape back
        project3D_grids = project3D_grids.view(batch_size, n_views, *project3D_grids.shape[1:])
        
        #merge all informations  
        project3D_grid = volume_aggregation_method(project3D_grids,self.volume_aggregation_cfg)
        
        #predict the 3D heatmap
        volume_heatmap_s1 ,volume_heatmap_s2,volume_heatmap_s3,vmax_cat,vmin_cat = self.volume_net(project3D_grid,body_center)   
        
        #predict final result
        output_s1 = integrate_tensor_3d_with_coordinates(volume_heatmap_s1)
        output_s2 = integrate_tensor_3d_with_coordinates(volume_heatmap_s2)
        output_s3 = integrate_tensor_3d_with_coordinates(volume_heatmap_s3)
        
        #concat the result from each stage
        batch = volume_heatmap_s1.shape[0]
        
        channel,tensor_size = output_s1.shape[1],output_s1.shape[2]
        output_s1 = output_s1.view(batch,1,channel,tensor_size)
        channel,tensor_size = output_s2.shape[1],output_s2.shape[2]
        output_s2 = output_s2.view(batch,1,channel,tensor_size)
        channel,tensor_size = output_s3.shape[1],output_s3.shape[2]
        output_s3 = output_s3.view(batch,1,channel,tensor_size)
        
        volume_heatmap_cat = [volume_heatmap_s1,volume_heatmap_s2,volume_heatmap_s3]
        
        output_cat = torch.cat((output_s1,output_s2,output_s3),1)
        
        return output_cat,volume_heatmap_cat,vmax_cat,vmin_cat

def get_pose_net(cfg,cfg_train):
    input_width,input_height = cfg_train.patch_width,cfg_train.patch_height    
    backbone_net = Feature2D_extracter_model(cfg.conv_2D_out_channels)    
    proj_unit = proj_splat(cfg)
    grid_conv3D_head = V2VModel(cfg.conv_2D_out_channels, cfg.pred_joints,cfg.box_range) 
    pose_net = PoseNet(input_width,input_height,backbone_net,proj_unit,cfg.volume_aggregation,grid_conv3D_head)
    return pose_net


def init_pose_net(pose_net, cfg):
    if os.path.exists(cfg.pretrained):
        pose_net_dict = pose_net.state_dict()                                 
        model = torch.load(cfg.pretrained)           
        pretrained_dict = {k: v for k, v in model.items() if k in pose_net_dict}            
        pose_net_dict.update(pretrained_dict)
        pose_net.load_state_dict(pose_net_dict)            
        print("Init Network from pretrained", cfg.pretrained)
      
