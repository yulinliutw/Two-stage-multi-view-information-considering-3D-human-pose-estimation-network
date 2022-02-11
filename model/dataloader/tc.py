from __future__ import print_function, division
import os
import torch
import pandas as pd
from pandas import Series, DataFrame
import h5py
import pickle
import linecache

from PIL import Image
import numpy as np

import scipy.io as sio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from spacepy import pycdf

#joint defination between each data#
'''
origin
 0'Hips',
 1'Spine',
 2'Spine1',
 3'Spine2',
 4'Spine3',
 5'Neck',
 6'Head',
 7'RightShoulder',
 8'RightArm',
 9'RightForeArm',
 10'RightHand',
 11'LeftShoulder',
 12'LeftArm',
 13'LeftForeArm',
 14'LeftHand',
 15'RightUpLeg',
 16'RightLeg',
 17'RightFoot',
 18'LeftUpLeg',
 19'LeftLeg',
 20'LeftFoot'
'''
tc_to_MPII =[17,16,15,18,19,20,0,3,5,6,10,9,8,12,13,14,6] 
# [RightFoot, Right knee, Right hip, Left hip, Left knee, LeftFoot, Hips, Spine, 
#Neck, Head Site, Right wrist, Right elbow, Right shoulder, Left shoulder, Left elbow, Left wrist, Head]
#camera id
cam_id = [1, 3, 5, 7]
#original image data size : H,W
tc_imgsize = [1080,1920]

class tcDataset(Dataset):
    def __init__(self,cfg,csvfile,cfg_net,transform = None):
        self.input_height = cfg_net.input_height
        self.input_width = cfg_net.input_width
        self.pred_joints = cfg_net.pred_joints
        self.root_dir = cfg.root_dir
        self.dataset_table = pd.read_csv(os.path.join(cfg.root_dir,csvfile))        
        self.cam_num = cfg.cam_num
        self.cam_matrix_path = cfg.cam_matrix_path            
        self.transform = transform
        self.rot_matrix = np.zeros((4,4))
        self.rot_matrix[0][0] = 1.0
        self.rot_matrix[1][2] = 1.0
        self.rot_matrix[2][1] = 1.0
        self.rot_matrix[3][3] = 1.0
        
        #get projection matrixs#                 
        proj_matrix = np.zeros((len(self.cam_num),3,4))        
        for i in range(len(self.cam_num)):
            intrinsic_matrix = np.zeros((3,3))
            camera_params = linecache.getline(os.path.join(self.root_dir,self.cam_matrix_path), (((cam_id[i])-1)*7+2)+1).strip()
            camera_params = np.array(list(map(float,camera_params.split())))
            intrinsic_matrix[:2, 2] = camera_params[2:4]
            intrinsic_matrix[0, 0] = camera_params[0]
            intrinsic_matrix[1, 1] = camera_params[1]
            intrinsic_matrix[2, 2] = 1.0
            
            extrinsic_matrix = np.zeros((3,4))
            camera_params = linecache.getline(os.path.join(self.root_dir,self.cam_matrix_path), (((cam_id[i])-1)*7+2)+3).strip()
            camera_params = np.array(list(map(float,camera_params.split())))
            extrinsic_matrix[0,0:3] = camera_params
            camera_params = linecache.getline(os.path.join(self.root_dir,self.cam_matrix_path), (((cam_id[i])-1)*7+2)+4).strip()
            camera_params = np.array(list(map(float,camera_params.split())))
            extrinsic_matrix[1,0:3] = camera_params
            camera_params = linecache.getline(os.path.join(self.root_dir,self.cam_matrix_path), (((cam_id[i])-1)*7+2)+5).strip()
            camera_params = np.array(list(map(float,camera_params.split())))
            extrinsic_matrix[2,0:3] = camera_params
            camera_params = linecache.getline(os.path.join(self.root_dir,self.cam_matrix_path), (((cam_id[i])-1)*7+2)+6).strip()
            camera_params = np.array(list(map(float,camera_params.split())))
            extrinsic_matrix[:,3] = camera_params
            extrinsic_matrix = np.dot(extrinsic_matrix,self.rot_matrix) #rotate the our coord to origin setting
            scale_matrix_input = np.zeros((4,4)) #scale the unit to m
            scale_matrix_input[0][0] = 0.001
            scale_matrix_input[1][1] = 0.001
            scale_matrix_input[2][2] = 0.001
            scale_matrix_input[3][3] = 1.0
            extrinsic_matrix = np.dot(extrinsic_matrix,scale_matrix_input)            
            
            scale_matrix = np.zeros((3,3))
            scale_matrix[0][0] = self.input_width/tc_imgsize[1]
            scale_matrix[1][1] = self.input_height/tc_imgsize[0]
            scale_matrix[2][2] = 1.0            
            intrinsic_matrix = np.dot(scale_matrix,intrinsic_matrix)            
            proj_matrix[i,0:3,0:4] = np.dot(intrinsic_matrix,extrinsic_matrix)
        self.proj_matrix_frame = proj_matrix
    def __len__(self):
        return len(self.dataset_table)
    def __getitem__(self,idx):
        #get images (numpy)#
        image_datas = {}
        for c in self.cam_num:
            image_path = os.path.join(self.root_dir,'A'+str(self.dataset_table.iloc[idx,3]),'S'+str(self.dataset_table.iloc[idx,2])
                                     ,'c'+str(cam_id[c-1]),str(self.dataset_table.iloc[idx,5]),'Color_'+str(self.dataset_table.iloc[idx,0])+'.bmp')
            image_data=Image.open(image_path) #current pixel value range : 0~255
            image_datas[c-1] = np.array(image_data)
        
        #get ground truth label#
        GT_path = os.path.join(self.root_dir,'A'+str(self.dataset_table.iloc[idx,3]),'GT','S'+str(self.dataset_table.iloc[idx,2]),self.dataset_table.iloc[idx,4],'gt_skel_gbl_pos.txt')        
        GT = linecache.getline(GT_path, int(self.dataset_table.iloc[idx,0]) + 1).strip()
        GT = np.array(list(map(float,GT.split())))
        GT = GT.reshape((21,3))
        GT = GT[tc_to_MPII,:] * 25.4 #uint : mm
        GT = np.transpose(np.dot(self.rot_matrix[0:3,0:3],np.transpose(GT))) #rotate the gt to our coord  
        GT[9,:] = (GT[16,:] * 2) - GT[8,:] #get head site               
        
        #form the sample#
        sample = {'image': image_datas,'camparam':self.proj_matrix_frame,'GT_label': GT}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


