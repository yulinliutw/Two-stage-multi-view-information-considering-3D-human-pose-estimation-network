from __future__ import print_function, division
import os
import torch
import pandas as pd
from pandas import Series, DataFrame
import h5py
import pickle

from PIL import Image
import numpy as np

import scipy.io as sio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from spacepy import pycdf

#joint defination between each data#
h36_to_MPII =[3,2,1,6,7,8,0,12,13,15,27,26,25,17,18,19,14] # RightFoot, Right knee, Right hip, Left hip, Left knee, LeftFoot, Hips, Spine 
#camera id                                                   Neck, Head Site, Right wrist, Right elbow, Right shoulder, Left shoulder, Left elbow, Left wrist, Head
cam_id = ['54138969', '55011271','58860488', '60457274']
#original image data size : H,W
h36m_imgsize = [1002,1000]

class h36Dataset(Dataset):
    def __init__(self,cfg,csvfile,cfg_net,transform = None):
        self.input_height = cfg_net.input_height
        self.input_width = cfg_net.input_width
        self.pred_joints = cfg_net.pred_joints
        self.root_dir = cfg.root_dir
        self.dataset_table = pd.read_csv(os.path.join(cfg.root_dir,csvfile))        
        self.cam_num = cfg.cam_num
        self.cam_matrix_path = cfg.cam_matrix_path            
        self.transform = transform
    def __len__(self):
        return len(self.dataset_table)
    def __getitem__(self,idx):
        #get images (numpy)#
        image_datas = {}
        for c in self.cam_num:
            image_path = os.path.join(self.root_dir,'A'+str(self.dataset_table.iloc[idx,4]),'S'+str(self.dataset_table.iloc[idx,3])
                                     ,'c'+str(cam_id[c-1]),str(self.dataset_table.iloc[idx,6]),'Color_'+str(self.dataset_table.iloc[idx,1])+'.bmp')
            image_data=Image.open(image_path) #current pixel value range : 0~255
            image_datas[c-1] = np.array(image_data)
        #get projection matrixs#
        cameras_params = h5py.File(os.path.join(self.root_dir,self.cam_matrix_path), 'r')          
        proj_matrix = np.zeros((len(self.cam_num),3,4))        
        for i in range(len(self.cam_num)):
            camera_params = cameras_params['subject'+str(self.dataset_table.iloc[idx,3])]['camera' + str(self.cam_num[i])] 
            intrinsic_matrix = np.zeros((3,3))
            intrinsic_matrix[:2, 2] = camera_params['c'][:, 0]
            intrinsic_matrix[0, 0] = camera_params['f'][0]
            intrinsic_matrix[1, 1] = camera_params['f'][1]
            intrinsic_matrix[2, 2] = 1.0
            extrinsic_matrix = np.zeros((3,4))
            extrinsic_matrix[0:3,0:3] = np.array(camera_params['R']).T
            extrinsic_matrix[:,3] = -np.array(camera_params['R']).T @ np.squeeze(camera_params['T'])
            scale_matrix = np.zeros((3,3))
            scale_matrix[0][0] = self.input_width/h36m_imgsize[1]
            scale_matrix[1][1] = self.input_height/h36m_imgsize[0]
            scale_matrix[2][2] = 1.0            
            intrinsic_matrix = np.dot(scale_matrix,intrinsic_matrix)            
            proj_matrix[i,0:3,0:4] = np.dot(intrinsic_matrix,extrinsic_matrix)
        proj_matrix_frame = proj_matrix
        #get ground truth label#
        GT_path = os.path.join(self.root_dir,'A'+str(self.dataset_table.iloc[idx,4]),'GT','gobal','S'+str(self.dataset_table.iloc[idx,3]),'MyPoseFeatures','D3_Positions',self.dataset_table.iloc[idx,0])    
        
        cdf_data = pycdf.CDF(GT_path)
        GTs_cdf = cdf_data.copy()
        for k in GTs_cdf.keys(): GTs = GTs_cdf[k][:]
        GTs = GTs[0,:,:]
        GTs = np.array(GTs, dtype=float) 
        GT = GTs[self.dataset_table.iloc[idx,1]-1,:]
        GT = GT.reshape((32,3))
        GT = GT[h36_to_MPII,:] #uint : mm
        cdf_data.close()
        
        #form the sample#
        sample = {'image': image_datas,'camparam':proj_matrix_frame,'GT_label': GT}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


