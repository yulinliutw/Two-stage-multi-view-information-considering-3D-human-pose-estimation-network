import torch
import torch.nn as nn
class proj_splat(nn.Module):
    def __init__(self,cfg):
        super(proj_splat, self).__init__()
        #default image size
        self.im_h = cfg.input_height
        self.im_w = cfg.input_width               
        
    def _gather_nd(self,feats, x, y):
       nR, fdim , fh, fw= feats.size()       
       nR, n = x.size()       
       sample_feats = torch.zeros(nR,fdim,n)       
       if(feats.is_cuda):           
          sample_feats = sample_feats.cuda()
       for idx in range(nR):
           sample_feats[idx,:,:] = feats[idx,:,y[idx,:].type(torch.LongTensor),x[idx,:].type(torch.LongTensor)].clone() 
       return sample_feats 
   
    def forward(self, feats, Kcam,vmax,vmin,nvox):   #the size of the Kcam should be n*v*3*4 , the size of body center should be n*3
        #ProjSplat
        nR, fdim , fh, fw= feats.size()
        batch,view,row,column = Kcam.size()
        
        rsz_h = float(fh) / self.im_h
        rsz_w = float(fw) / self.im_w   
            
        #Project grid     
        
        rs_grid_batch = torch.zeros(batch,4,nvox**3)
        
        for batch_n in range(batch):
            #Create voxel grid            
            grid_range_x = torch.linspace(vmin[batch_n,0], vmax[batch_n,0], nvox)
            grid_range_y = torch.linspace(vmin[batch_n,1], vmax[batch_n,1], nvox)
            grid_range_z = torch.linspace(vmin[batch_n,2], vmax[batch_n,2], nvox)
            grid = torch.stack(
                        torch.meshgrid(grid_range_x, grid_range_y, grid_range_z))
            rs_grid = torch.reshape(grid, [3, -1])
            nV = rs_grid.size()[1]
            rs_grid = torch.cat([rs_grid, torch.ones([1, nV])], 0)
            rs_grid_batch[batch_n,:,:] = rs_grid
            
        if(feats.is_cuda): #if feature map is stored in the gpu, move the rs_grid to gpu
            rs_grid_batch = rs_grid_batch.cuda()
            
        #World2Cam
        
        im_p = torch.bmm(torch.reshape(Kcam, [batch,view*row, 4]), rs_grid_batch)        
        im_p = torch.reshape(im_p,[batch*view,3,-1]) #reshape the im_p to nR samples with shape 3 * (nvox^3)        
        im_x, im_y, im_z = im_p[:,0, :], im_p[:,1, :], im_p[:,2, :] #get each data's sample index
        im_x = (im_x / im_z) * rsz_w #size: nR x (nvox^3)
        im_y = (im_y / im_z) * rsz_h #size: nR x (nvox^3)       
    
        #Bilinear interpolation        
        im_x = torch.clamp(im_x, 0, fw - 1)
        im_y = torch.clamp(im_y, 0, fh - 1)
        im_x0 = torch.floor(im_x)        
        im_x1 = im_x0 + 1  
        im_x1 = torch.clamp(im_x1, 0, fw - 1)
        im_y0 = torch.floor(im_y)     
        im_y1 = im_y0 + 1  
        im_y1 = torch.clamp(im_y1, 0, fh - 1)
        im_x0_f, im_x1_f = im_x0.float(), im_x1.float()
        im_y0_f, im_y1_f = im_y0.float(), im_y1.float()
    
        #Gather  values       
        Ia = self._gather_nd(feats, im_x0, im_y0)
        Ib = self._gather_nd(feats, im_x0, im_y1)
        Ic = self._gather_nd(feats, im_x1, im_y0)
        Id = self._gather_nd(feats, im_x1, im_y1)
    
        #Calculate bilinear weights
        wa = (im_x1_f - im_x) * (im_y1_f - im_y)
        wb = (im_x1_f - im_x) * (im_y - im_y0_f)
        wc = (im_x - im_x0_f) * (im_y1_f - im_y)
        wd = (im_x - im_x0_f) * (im_y - im_y0_f)
        Ibilin = torch.zeros(nR, fdim, nvox**3)
        if(feats.is_cuda):
            Ibilin = Ibilin.cuda()          
        for idx in range(nR):
           Ibilin[idx,:,:] = ((wa[idx,:] * Ia[idx,:,:]) + (wb[idx,:] * Ib[idx,:,:]) + (wc[idx,:] * Ic[idx,:,:]) + (wd[idx,:] * Id[idx,:,:]))     
        Ibilin = torch.reshape(Ibilin, 
             [nR, fdim, nvox, nvox, nvox])    
        return Ibilin


