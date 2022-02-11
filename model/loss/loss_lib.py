import torch
import torch.nn as nn

def label_normorlize(label,body_center,cfg_label):
    box_range = cfg_label.box_range
    vmin = body_center - box_range
    vmax = body_center + box_range
    mean_label = ((vmax)+(vmin))/2
    scale_label = ((vmax)-(vmin))/2
    
    label[:,:,0] = (label[:,:,0] - mean_label[:,0].view(-1,1)) / scale_label[:,0].view(-1,1)
    label[:,:,1] = (label[:,:,1] - mean_label[:,1].view(-1,1)) / scale_label[:,1].view(-1,1)
    label[:,:,2] = (label[:,:,2] - mean_label[:,2].view(-1,1)) / scale_label[:,2].view(-1,1)
    
    return label

def label_normorlize_v2(label,vmax,vmin):
    mean_label = ((vmax)+(vmin))/2
    scale_label = ((vmax)-(vmin))/2
    label_norm = torch.zeros(label.shape[0],label.shape[1],label.shape[2])
    if(label.is_cuda):
        label_norm = label_norm.cuda()
    label_norm[:,:,0] = (label[:,:,0] - mean_label[:,0].view(-1,1)) / scale_label[:,0].view(-1,1)
    label_norm[:,:,1] = (label[:,:,1] - mean_label[:,1].view(-1,1)) / scale_label[:,1].view(-1,1)
    label_norm[:,:,2] = (label[:,:,2] - mean_label[:,2].view(-1,1)) / scale_label[:,2].view(-1,1)
    
    return label_norm

def Grid_3D_loss_l2(predict,label):
    loss_func = nn.MSELoss()
    loss = loss_func(predict,label)
    return loss
    
def Grid_3D_loss_crossEntropy(predict,label):
    loss_func = nn.BCELoss()
    loss = loss_func(predict,label)
    return loss   

def L1_loss(predict,label,body_center,cfg_label):
    loss_func = nn.L1Loss(reduction='sum')
    label = label_normorlize(label,body_center,cfg_label)
    loss = loss_func(predict,label)
    return loss    
class bones_angle_loss_multi_stage(nn.Module):
    def __init__(self,beta = 0.0001):
        super(bones_angle_loss_multi_stage,self).__init__()
        self.beta = beta
        self.parent_joint_list = [13,12,14,11,3,2,4,1]
        self.two_neighbor_joint_list = [[8,14],[8,11],[13,15],[12,10],[6,4],[6,1],[3,5],[2,0]]
        self.l1_loss_func = nn.L1Loss(reduction='sum') 
    def label_normorlize_v2(self,label,vmax,vmin):
        mean_label = ((vmax)+(vmin))/2
        scale_label = ((vmax)-(vmin))/2
        label_norm = torch.zeros(label.shape[0],label.shape[1],label.shape[2])
        if(label.is_cuda):
            label_norm = label_norm.cuda()    
        label_norm[:,:,0] = (label[:,:,0] - mean_label[:,0].view(-1,1)) / scale_label[:,0].view(-1,1)
        label_norm[:,:,1] = (label[:,:,1] - mean_label[:,1].view(-1,1)) / scale_label[:,1].view(-1,1)
        label_norm[:,:,2] = (label[:,:,2] - mean_label[:,2].view(-1,1)) / scale_label[:,2].view(-1,1)        
        return label_norm     
    def forward(self,predict_cat,label,vmax_cat,vmin_cat):
        stage = len(predict_cat)
        loss = 0.0
        for idx in range(stage):
            label_norm = self.label_normorlize_v2(label,vmax_cat[:,idx,:],vmin_cat[:,idx,:])
            if((label_norm.max()<=1)and(label_norm.max()>=-1)and(label_norm.min()<=1)and(label_norm.min()>=-1)):
                batch_size = predict_cat[idx].shape[0]        
                for i in range(len(self.parent_joint_list)):
                    #caculate the cosine value between two bone vector in predict
                    first_bone_predict = predict_cat[idx][:,self.two_neighbor_joint_list[i][0],:] - predict_cat[idx][:,self.parent_joint_list[i],:]
                    second_bone_predict = predict_cat[idx][:,self.two_neighbor_joint_list[i][1],:] - predict_cat[idx][:,self.parent_joint_list[i],:]
                    first_bone_predict = first_bone_predict + 0.0000000001
                    second_bone_predict = second_bone_predict + 0.0000000001
                    first_bone_predict_length = (((first_bone_predict**2).sum(1))**0.5).view(batch_size,1)
                    second_bone_predict_length = (((second_bone_predict**2).sum(1))**0.5).view(batch_size,1)
                    first_bone_predict = first_bone_predict / first_bone_predict_length
                    second_bone_predict = second_bone_predict / second_bone_predict_length
                    cosine_two_bone_predict = (first_bone_predict * second_bone_predict).sum(1)
                    
                   #caculate the cosine value between two bone vector in label
                    first_bone_label = label_norm[:,self.two_neighbor_joint_list[i][0],:] - label_norm[:,self.parent_joint_list[i],:]
                    second_bone_label = label_norm[:,self.two_neighbor_joint_list[i][1],:] - label_norm[:,self.parent_joint_list[i],:]
                    first_bone_label= first_bone_label + 0.0000000001
                    second_bone_label = second_bone_label + 0.0000000001
                    first_bone_label_length = (((first_bone_label**2).sum(1))**0.5).view(batch_size,1)
                    second_bone_label_length = (((second_bone_label**2).sum(1))**0.5).view(batch_size,1)
                    first_bone_label = first_bone_label / first_bone_label_length
                    second_bone_label = second_bone_label / second_bone_label_length
                    cosine_two_bone_label = (first_bone_label * second_bone_label).sum(1)
                    
                    #get loss
                    loss += self.beta * self.l1_loss_func(cosine_two_bone_predict,cosine_two_bone_label)
            
        return loss
    
class bone_length_loss_multi_stage(nn.Module):
    def __init__(self,cfg_label,beta = 0.001):
        super(bone_length_loss_multi_stage,self).__init__()
        self.joint_num = cfg_label.pred_joints
        self.beta = beta
        #generate joint pair idx 
        self.joint_pair_idx = []
        for i in range(self.joint_num - 1):
            child_joint_list = []
            for v in range(i+1,self.joint_num,1):               
                child_joint_list.append(v)
            self.joint_pair_idx.append(child_joint_list) 
        self.l1_loss_func = nn.L1Loss(reduction='sum')  
    def label_normorlize_v2(self,label,vmax,vmin):
        mean_label = ((vmax)+(vmin))/2
        scale_label = ((vmax)-(vmin))/2
        label_norm = torch.zeros(label.shape[0],label.shape[1],label.shape[2])
        if(label.is_cuda):
            label_norm = label_norm.cuda()    
        label_norm[:,:,0] = (label[:,:,0] - mean_label[:,0].view(-1,1)) / scale_label[:,0].view(-1,1)
        label_norm[:,:,1] = (label[:,:,1] - mean_label[:,1].view(-1,1)) / scale_label[:,1].view(-1,1)
        label_norm[:,:,2] = (label[:,:,2] - mean_label[:,2].view(-1,1)) / scale_label[:,2].view(-1,1)        
        return label_norm     
    def forward(self,predict_cat,label,vmax_cat,vmin_cat):        
        stage = len(predict_cat)
        loss = 0.0
        for idx in range(stage):
            label_norm = self.label_normorlize_v2(label,vmax_cat[:,idx,:],vmin_cat[:,idx,:])
            if((label_norm.max()<=1)and(label_norm.max()>=-1)and(label_norm.min()<=1)and(label_norm.min()>=-1)):
                for j in range(self.joint_num-1):
                    loss +=  self.beta * self.l1_loss_func((((predict_cat[idx][:,[j],:] - predict_cat[idx][:,self.joint_pair_idx[j],:])**2).sum(2))**0.5,(((label_norm[:,[j],:] - label_norm[:,self.joint_pair_idx[j],:])**2).sum(2))**0.5)        
        return loss   

class L1_loss_multi_stage(nn.Module):
    def __init__(self):
        super(L1_loss_multi_stage,self).__init__() 
        
    def label_normorlize_v2(self,label,vmax,vmin):
        mean_label = ((vmax)+(vmin))/2
        scale_label = ((vmax)-(vmin))/2        
        label_norm = torch.zeros(label.shape[0],label.shape[1],label.shape[2])
        if(label.is_cuda):
            label_norm = label_norm.cuda()    
        label_norm[:,:,0] = (label[:,:,0] - mean_label[:,0].view(-1,1)) / scale_label[:,0].view(-1,1)
        label_norm[:,:,1] = (label[:,:,1] - mean_label[:,1].view(-1,1)) / scale_label[:,1].view(-1,1)
        label_norm[:,:,2] = (label[:,:,2] - mean_label[:,2].view(-1,1)) / scale_label[:,2].view(-1,1)
        
        return label_norm    
        
    def forward(self,predict_cat,label,vmax_cat,vmin_cat):
        stage = len(predict_cat)
        loss = 0.0        
        #stage n  
        for idx in range(stage):        
            label_norm = self.label_normorlize_v2(label,vmax_cat[:,idx,:],vmin_cat[:,idx,:])
            if((label_norm.max()<=1)and(label_norm.max()>=-1)and(label_norm.min()<=1)and(label_norm.min()>=-1)):
                loss += torch.sum(torch.abs(predict_cat[idx]-label_norm)) #l1 loss
            else:                            
                print("out_of_bound")
           
        return loss

class VolumetricCELoss(nn.Module):
    def __init__(self,cfg_label,beta = 0.01):
        super().__init__()
        self.box_range = cfg_label.box_range
        self.nvox = cfg_label.nvox
        self.beta = beta

    def forward(self,volumes_batch_pred, label,body_center):
        batch_size, predict_joint, x_size, y_size, z_size = volumes_batch_pred.shape
        loss = 0.0
        n_losses = 0
        volumes_batch_pred = volumes_batch_pred.reshape((batch_size, predict_joint, -1))
        volumes_batch_pred = nn.functional.softmax(volumes_batch_pred, dim=2)
        volumes_batch_pred = volumes_batch_pred.reshape((batch_size,  predict_joint, x_size, y_size, z_size))
        
        vmin = body_center - self.box_range
        vmax = body_center + self.box_range
        mean_label = ((vmax)+(vmin))/2
        scale_label = ((vmax)-(vmin))/2
        for batch_i in range(batch_size):
            GT_label = label[batch_i,:,:]
            GT_label[:,0] = (GT_label[:,0] - mean_label[batch_i,0]) / scale_label[batch_i,0]
            GT_label[:,1] = (GT_label[:,1] - mean_label[batch_i,1]) / scale_label[batch_i,1]
            GT_label[:,2] = (GT_label[:,2] - mean_label[batch_i,2]) / scale_label[batch_i,2] 
            GT_label_grid_idx = GT_label	
            GT_label_grid_idx = torch.floor(((GT_label_grid_idx +1)/2) * (self.nvox - 1)).type(torch.LongTensor)
            for joint_i in range(predict_joint):
                loss += self.beta * (-torch.log(volumes_batch_pred[batch_i, joint_i, GT_label_grid_idx[joint_i,0], GT_label_grid_idx[joint_i,1], GT_label_grid_idx[joint_i,2]] + 1e-6))
                n_losses += 1 		

        return loss / n_losses
    
class VolumetricCELoss_multi_stage(nn.Module):
    def __init__(self,beta = 0.01):
        super().__init__()        
        self.beta = beta

    def forward(self,volumes_batch_pred_cat, label,vmax_cat,vmin_cat):
        stage = len(volumes_batch_pred_cat)
        total_loss = 0.0  
        
        for stage_idx in range(stage):
            batch_size, predict_joint, x_size, y_size, z_size = volumes_batch_pred_cat[stage_idx].shape
            loss = 0.0
            n_losses = 0
            
            volumes_batch_pred = volumes_batch_pred_cat[stage_idx].reshape((batch_size, predict_joint, -1))
            volumes_batch_pred = nn.functional.softmax(volumes_batch_pred, dim=2)
            volumes_batch_pred = volumes_batch_pred.reshape((batch_size,  predict_joint, x_size, y_size, z_size))
            
            vmin = vmin_cat[:,[stage_idx],:]
            vmax = vmax_cat[:,[stage_idx],:]
            mean_label = ((vmax)+(vmin))/2
            scale_label = ((vmax)-(vmin))/2
            
            GT_label = label[:,:,:].clone()
            GT_label[:,:,0] = (GT_label[:,:,0] - mean_label[:,[0],0]) / scale_label[:,[0],0]
            GT_label[:,:,1] = (GT_label[:,:,1] - mean_label[:,[0],1]) / scale_label[:,[0],1]
            GT_label[:,:,2] = (GT_label[:,:,2] - mean_label[:,[0],2]) / scale_label[:,[0],2] 
            GT_label_grid_idx = GT_label	
            GT_label_grid_idx = torch.floor(((GT_label_grid_idx +1)/2) * (x_size - 1)).type(torch.LongTensor)
            
            if((GT_label_grid_idx.max()<(x_size))and(GT_label_grid_idx.max()>0)and(GT_label_grid_idx.min()<(x_size))and(GT_label_grid_idx.min()>0)):
                for batch_i in range(batch_size):                
                    for joint_i in range(predict_joint):                    
                        loss += self.beta * (-torch.log(volumes_batch_pred[batch_i, joint_i, GT_label_grid_idx[batch_i,joint_i,0], GT_label_grid_idx[batch_i,joint_i,1], GT_label_grid_idx[batch_i,joint_i,2]] + 1e-6))
                        n_losses += 1 		
                total_loss = total_loss + (loss / n_losses)
            else:
                print("out_of_bound")
                loss += 0 * volumes_batch_pred[:,:,:].sum()
                total_loss = total_loss + loss
        return total_loss
    
