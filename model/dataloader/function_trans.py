from torchvision import transforms, utils
import torch
     
class Normalize(object):
    """Normalize the img and label in tensor"""
    def __init__(self,dataset,cfg_img,cfg_im_wh):
        self.dataset = dataset
        self.mean_img = cfg_img.mean
        self.std_img = cfg_img.std
        self.im_h = cfg_im_wh.patch_height
        self.im_w = cfg_im_wh.patch_width
    def __call__(self, sample):  
        GT_label, camparam, image = sample['GT_label'], sample['camparam'], sample['image']
        image_norm = torch.zeros(len(image),3,self.im_h,self.im_w)
        for img_idx in range(len(image)):
            image_norm[img_idx,:,:,:] = transforms.Normalize(self.mean_img,self.std_img)(image[img_idx])
        return {
                'GT_label': GT_label,
                'camparam': camparam,                
                'image': image_norm
                }
        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self,dataset):
        self.dataset = dataset
    def __call__(self, sample):
       GT_label, camparam, image = sample['GT_label'], sample['camparam'], sample['image'] 
       image_tensor = {}
       for img_idx in range(len(image)):
           image_tensor[img_idx] = transforms.ToTensor()(image[img_idx])
       return {
                'GT_label': torch.from_numpy(GT_label), #size: j x3
                'camparam': torch.from_numpy(camparam), #size: v x 3 x4                
                'image': image_tensor #size: v x 3 x h x w
                } 

