# Reference: https://github.com/dragonbook/V2V-PoseNet-pytorch

import torch.nn as nn
import torch.nn.functional as F
import torch
from .operator_function import NearstIntepolation,volume_aggregation_method,volume_viewBased_fetureenforcememt
from common.utility.visualization import *


class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)


class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)


class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample3DBlock, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self,input_channels):
        super().__init__()
        #encoder bone
        self.encoder_basic1 = Basic3DBlock(input_channels, 32, 7)
        self.encoder_res2 = Res3DBlock(32, 32)
        self.encoder_res3 = Res3DBlock(32, 32)
        
        self.encoder_pool4 = Pool3DBlock(2)
        self.encoder_res4 = Res3DBlock(32, 64)        
        
        self.encoder_pool5 = Pool3DBlock(2)
        self.encoder_res5 = Res3DBlock(64, 128)

        self.encoder_pool6 = Pool3DBlock(2)
        self.encoder_res6 = Res3DBlock(128, 128)
        
        #extract multi-scale structure 
        self.conv_16x16x16 = nn.Sequential(
                            nn.Conv3d(64, 32, kernel_size=1, stride=1, padding=0),
                            nn.Upsample(scale_factor = 2, mode='trilinear')
                            )
        self.conv_8x8x8 = nn.Sequential(
                            nn.Conv3d(128, 32, kernel_size=1, stride=1, padding=0),
                            nn.Upsample(scale_factor = 4, mode='trilinear')
                            )
        self.conv_4x4x4 = nn.Sequential(
                            nn.Conv3d(128, 32, kernel_size=1, stride=1, padding=0),
                            nn.Upsample(scale_factor = 8, mode='trilinear')
                            )
        
    def forward(self, x):
        #encoder bone
        x_1 = self.encoder_basic1(x)
        x_1 = self.encoder_res2(x_1)
        x_1 = self.encoder_res3(x_1)
        
        x_2 = self.encoder_pool4(x_1)
        x_2 = self.encoder_res4(x_2)
        
        x_3 = self.encoder_pool5(x_2)
        x_3 = self.encoder_res5(x_3)
        
        x_4 = self.encoder_pool6(x_3)
        x_4 = self.encoder_res6(x_4)
        
        #extract multi-scale structure 
        feature_16x16x16 = self.conv_16x16x16(x_2)        
        
        feature_8x8x8 = self.conv_8x8x8(x_3)
        feature_4x4x4 = self.conv_4x4x4(x_4)
        
        feature_cat = torch.cat([ x_1 + feature_16x16x16, feature_8x8x8 + feature_4x4x4],1)        
        
        return feature_cat
    
class Decorder(nn.Module):
    def __init__(self,input_channels,output_channels):
        super().__init__()
        self.decoder_res1 = Res3DBlock(input_channels, 64)
        self.decoder_basic2 = Basic3DBlock(64, 32, 1)
        self.decoder_out = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):        
        x = self.decoder_res1(x)
        x = self.decoder_basic2(x)
        x = self.decoder_out(x)
        
        return x

class Decorder_v2(nn.Module):
    def __init__(self,input_channels,output_channels):
        super().__init__()
        self.decoder_res1 = Res3DBlock(input_channels, input_channels)                
        self.decoder_basic2 = Basic3DBlock(input_channels, 64, 1)
        self.decoder_out = nn.Conv3d(64, output_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):        
        x = self.decoder_res1(x)        
        x = self.decoder_basic2(x)
        x = self.decoder_out(x)
        
        return x    

class crop_predict(nn.Module):
    def __init__(self,boundry_offect):  
        super().__init__()
        self.boundry_offect = boundry_offect
        self.nearst_intepolation = NearstIntepolation()
        
    def forward(self,heatmap,vmin_s1,vmax,vmin):         
        #use heatmap to find the new boundry
        batch_size,joint_num,volume_shape_heatmap = heatmap.shape[0],heatmap.shape[1],heatmap.shape[2]        
        per_joint_position = torch.zeros(batch_size,joint_num,3)     
                    
        #find per joint position
        for batch_idx in range(batch_size):
            for joint_idx in range(joint_num):
                joint_position = (heatmap[batch_idx,joint_idx,:]==torch.max(heatmap[batch_idx,joint_idx,:])).nonzero().float()                
                per_joint_position[batch_idx,joint_idx,:] = torch.mean(joint_position,0)
                
        #find max min boundry for each batch & extend it a little        
        max_boundry = torch.zeros(batch_size,1,3)
        min_boundry = torch.zeros(batch_size,1,3)
        
        if(heatmap.is_cuda):
             per_joint_position = per_joint_position.cuda()
             max_boundry = max_boundry.cuda()
             min_boundry = min_boundry.cuda()
        for batch_idx in range(batch_size):
            #find max & min boundry for x dim
            max_boundry[batch_idx,0,0] = torch.max(per_joint_position[batch_idx,:,0])
            min_boundry[batch_idx,0,0] = torch.min(per_joint_position[batch_idx,:,0])
            #find max & min boundry for y dim
            max_boundry[batch_idx,0,1] = torch.max(per_joint_position[batch_idx,:,1])
            min_boundry[batch_idx,0,1] = torch.min(per_joint_position[batch_idx,:,1])          
            #find max & min boundry for z dim
            max_boundry[batch_idx,0,2] = torch.max(per_joint_position[batch_idx,:,2])
            min_boundry[batch_idx,0,2] = torch.min(per_joint_position[batch_idx,:,2])
            
        #extend it a little
        max_boundry[:,:,:] = torch.clamp(max_boundry + self.boundry_offect,min = 0.0, max = volume_shape_heatmap -1)
        min_boundry[:,:,:] = torch.clamp(min_boundry - self.boundry_offect,min = 0.0, max = volume_shape_heatmap -1)

        #get interval in real world for current heatmap
        interval_real_heatmap = (vmax - vmin) / (volume_shape_heatmap -1)        
        
        #get new max min boundry in real world for next heatmap
        max_boundry = vmin + ((max_boundry/ (volume_shape_heatmap -1))  * (vmax - vmin))
        min_boundry = vmin + ((min_boundry/ (volume_shape_heatmap -1))  * (vmax - vmin))
        
        rs_grid_batch_heatmap = torch.zeros(batch_size,(volume_shape_heatmap * 2) ** 3,3)        
        if(heatmap.is_cuda):            
            rs_grid_batch_heatmap = rs_grid_batch_heatmap.cuda()
        
        #use new max min boundry in real world to create the sample space for next feature map
        #get sample position in real world
        for batch_n in range(batch_size):
            #Create voxel grid             
            grid_range_x = torch.linspace(min_boundry[batch_n,0,0], max_boundry[batch_n,0,0], volume_shape_heatmap * 2)
            grid_range_y = torch.linspace(min_boundry[batch_n,0,1], max_boundry[batch_n,0,1], volume_shape_heatmap * 2)
            grid_range_z = torch.linspace(min_boundry[batch_n,0,2], max_boundry[batch_n,0,2], volume_shape_heatmap * 2)
            grid = torch.stack(torch.meshgrid(grid_range_x, grid_range_y, grid_range_z))         
            rs_grid = torch.reshape(grid, [3, -1])
            rs_grid = rs_grid.t()
            if(heatmap.is_cuda):
                rs_grid = rs_grid.cuda()           
            #for heatmap        
            rs_grid_batch_heatmap[batch_n,:,:] = (rs_grid - vmin[batch_n,:,:]) / interval_real_heatmap[batch_n,:,:]            
           
        #sample the information from feature map and heatmap        
        #interpolated_data from heatmap
        interpolated_data_heatmap = self.nearst_intepolation(heatmap,rs_grid_batch_heatmap) 
        interpolated_data_heatmap = interpolated_data_heatmap.view(batch_size,joint_num,volume_shape_heatmap * 2,volume_shape_heatmap * 2,volume_shape_heatmap * 2)
        
        return interpolated_data_heatmap,max_boundry,min_boundry            
        

class V2VModel(nn.Module):
    def __init__(self,proj_unit, nvox, input_channels, output_channels, box_range ,volume_aggregation_cfg):
        super().__init__()
        #get initial box range
        self.box_range = box_range 
        #get final 2d to 3d sample volume size
        self.nvox = nvox
        #get 2d to 3d feature sampler
        self.proj_unit = proj_unit  
        
        #get 3d volume feature extractor
        self.feature_extractor = Encoder(input_channels)
        self.feature_extractor_2 = Encoder(input_channels)
        self.downsample = Pool3DBlock(2)             
        
        #multi stage prediction
        self.volume_viewBased_fetureenforcememt_module = volume_viewBased_fetureenforcememt()
        self.volume_aggregation_cfg = volume_aggregation_cfg
        
        self.predict_stage1 = Decorder(64,output_channels)
        self.crop_predict = crop_predict(2)        
        self.predict_stage_2 = Decorder_v2(64 + output_channels,output_channels)       
        
        self._initialize_weights()
        
        
        
    def forward(self,color_streams,Kcam, body_center):
        #stage 1
        #get stage 1 boundry
        vmax_s1 = body_center + self.box_range 
        vmin_s1 = body_center - self.box_range 
        vmax_s1 = vmax_s1.view(Kcam.shape[0],1,3)
        vmin_s1 = vmin_s1.view(Kcam.shape[0],1,3)
        
        #project the 2d features to form a 3d feature        
        project3D_grids_s1 = self.proj_unit(color_streams,Kcam,vmax_s1[:,0,:],vmin_s1[:,0,:],self.nvox)               
        # reshape back
        project3D_grids_s1 = project3D_grids_s1.view(Kcam.shape[0], Kcam.shape[1], *project3D_grids_s1.shape[1:])        
          
        #for each view, other view info. help target view to understand which channel should be pay attention
        project3D_grids_enformenct_s1 = self.volume_viewBased_fetureenforcememt_module(project3D_grids_s1)
        
        #merge all informations  
        project3D_grid_s1 = volume_aggregation_method(project3D_grids_enformenct_s1,self.volume_aggregation_cfg)        
        
        #extract the multi scale feature
        feature_16x16x16 = self.feature_extractor(project3D_grid_s1)
        feature_16x16x16 = self.downsample(feature_16x16x16)               
        predict_16x16x16 = self.predict_stage1(feature_16x16x16)
        
        #stage 2 
        #get stage 2 boundry and crop the previous prediction              
        predict_crop,vmax_s2,vmin_s2 = self.crop_predict(predict_16x16x16,vmin_s1,vmax_s1,vmin_s1)        
        vmax_s2 = vmax_s2.view(Kcam.shape[0],1,3)
        vmin_s2 = vmin_s2.view(Kcam.shape[0],1,3)
        
        #project the 2d features to form a 3d feature        
        project3D_grids_s2 = self.proj_unit(color_streams,Kcam,vmax_s2[:,0,:],vmin_s2[:,0,:],self.nvox)               
        # reshape back
        project3D_grids_s2 = project3D_grids_s2.view(Kcam.shape[0], Kcam.shape[1], *project3D_grids_s2.shape[1:])        
          
        #for each view, other view info. help target view to understand which channel should be pay attention
        project3D_grids_enformenct_s2 = self.volume_viewBased_fetureenforcememt_module(project3D_grids_s2)
        
        #merge all informations  
        project3D_grid_s2 = volume_aggregation_method(project3D_grids_enformenct_s2,self.volume_aggregation_cfg)        
        
        #extract the multi scale feature
        feature_32x32x32 = self.feature_extractor_2(project3D_grid_s2)        
        #concat the multi scale feature and previous prediction
        feature_32x32x32 = torch.cat((feature_32x32x32,predict_crop),1)              
        predict_32x32x32 = self.predict_stage_2(feature_32x32x32)
        
        #get the min max boundry in each stage and concact them               
        vmax_cat = torch.cat((vmax_s1,vmax_s2),1)
        vmin_cat = torch.cat((vmin_s1,vmin_s2),1)
        
        return [predict_16x16x16,predict_32x32x32],vmax_cat,vmin_cat,[feature_16x16x16,feature_32x32x32]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
