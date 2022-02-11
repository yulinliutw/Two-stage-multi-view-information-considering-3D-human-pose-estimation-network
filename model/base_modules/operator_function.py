import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import matplotlib.cm
import numpy as np
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure

def find_3Dbody_center_candidate(bdcenter_2d_nview,camParam,unit = 'mm'): #size-> bdcenter_2d_nview : batch,view,2 ; camParam: batch,view,3,4
    bdcenter_2d_nview = bdcenter_2d_nview.cpu()
    camParam = camParam.cpu()
    batch_size, view, coord = bdcenter_2d_nview.shape
    total_view = batch_size * view
    
    #parameter
    U = torch.zeros(3,total_view) 
    u_2 = torch.zeros(1,total_view) 
    A = torch.zeros(1,total_view) 
    B = torch.zeros(1,total_view) 
    C = torch.zeros(1,total_view) 
    I = torch.zeros(1,total_view)
    J = torch.zeros(1,total_view)
    K = torch.zeros(1,total_view)
    N = torch.zeros(1,total_view)
    M = torch.zeros(1,total_view)
    L = torch.zeros(1,total_view)
    O = torch.zeros(1,total_view)
    P = torch.zeros(1,total_view)
    Q = torch.zeros(1,total_view)
    Matrix = torch.zeros(batch_size,3,3)
    right_vactor = torch.zeros(batch_size,3,1)
    
    unit_scaling = 0
    if(unit == 'mm'):
       unit_scaling = 1000
    else:
        unit_scaling = 1        
    
    #get inverse camParam
    camParam = camParam.view(-1,*camParam.shape[2:])
    inv_camParam = torch.zeros(total_view,4,4)
    inv_camParam[:,0:3,:] = camParam
    inv_camParam[:,3,3] = 1
    inv_camParam = torch.inverse(inv_camParam)
    
    #get point T for each view
    PT=torch.zeros(3,total_view)
    p_4d_T=torch.ones(4,total_view)
    p_4d_T[0:2,:] = bdcenter_2d_nview.view(total_view,2).t()
    p_4d_T[0:3,:] = p_4d_T[0:3,:] * unit_scaling
    p_4d_T = torch.einsum("vre, ev -> vr",inv_camParam,p_4d_T) # v : view , r & e : size of camParam , e also the element in one point
    PT = p_4d_T.t()[0:3,:]
    #get point P for each view
    PP=torch.zeros(3,total_view)
    p_4d_T=torch.ones(4,total_view)
    p_4d_T[0:2,:] = bdcenter_2d_nview.view(total_view,2).t()
    p_4d_T[0:3,:] = p_4d_T[0:3,:] * (unit_scaling * 4)
    p_4d_T = torch.einsum("vre, ev -> vr",inv_camParam,p_4d_T) # v : view , r & e : size of camParam , e also the element in one point
    PP = p_4d_T.t()[0:3,:]
    
    #get U vector to describe the projection line in each view
    U[:,:] =  PP - PT
    #Solove the equation to get the 3D body center from 2D points
    u_2[0,:] = torch.sum(U * U , 0)
    
    A[0,:] = (PP[1,:] * PT[2,:]) - (PP[2,:] * PT[1,:])
    B[0,:] = (PP[2,:] * PT[0,:]) - (PP[0,:] * PT[2,:])
    C[0,:] = (PP[0,:] * PT[1,:]) - (PP[1,:] * PT[0,:])
    
    I[0,:] = (U[2,:]**2 + U[1,:]**2) / u_2[0,:]
    J[0,:] = (U[2,:]**2 + U[0,:]**2) / u_2[0,:]
    K[0,:] = (U[1,:]**2 + U[0,:]**2) / u_2[0,:]
    
    N[0,:] = (-2*(U[2,:]*U[1,:])) / u_2[0,:]
    M[0,:] = (-2*(U[0,:]*U[2,:])) / u_2[0,:]
    L[0,:] = (-2*(U[1,:]*U[0,:])) / u_2[0,:]
    
    O[0,:]=((2 * U[2,:] * B[0,:])-(2 * U[1,:] * C[0,:])) / u_2[0,:]
    P[0,:]=((2 * U[0,:] * C[0,:])-(2 * U[2,:] * A[0,:])) / u_2[0,:]
    Q[0,:]=((2 * U[1,:] * A[0,:])-(2 * U[0,:] * B[0,:])) / u_2[0,:]
    
    i = torch.sum(I.view(1, batch_size, view),2)
    j = torch.sum(J.view(1, batch_size, view),2)
    k = torch.sum(K.view(1, batch_size, view),2)
    n = torch.sum(N.view(1, batch_size, view),2)
    m = torch.sum(M.view(1, batch_size, view),2)
    l = torch.sum(L.view(1, batch_size, view),2)
    o = torch.sum(O.view(1, batch_size, view),2)
    p = torch.sum(P.view(1, batch_size, view),2)
    q = torch.sum(Q.view(1, batch_size, view),2)
    
    Matrix[:,0,0] = i[0,:] * 2
    Matrix[:,0,1] = l[0,:] * -1
    Matrix[:,0,2] = m[0,:] * -1
    
    Matrix[:,1,0] = l[0,:] * -1
    Matrix[:,1,1] = j[0,:] * 2
    Matrix[:,1,2] = n[0,:] * -1
    
    Matrix[:,2,0] = m[0,:] * -1
    Matrix[:,2,1] = n[0,:] * -1
    Matrix[:,2,2] = k[0,:] * 2
    
    right_vactor[:,0,0] = o[0,:]
    right_vactor[:,1,0] = p[0,:]
    right_vactor[:,2,0] = q[0,:]
    
    body_center = torch.bmm(torch.inverse(Matrix),right_vactor)
    return body_center
class NearstIntepolation(nn.Module):
    def __init__(self):
        super(NearstIntepolation, self).__init__()

    def forward(self, input_feats, sampling_grid):
        batch,channel = input_feats.shape[0],input_feats.shape[1]
        sample_features = torch.zeros(batch,channel,sampling_grid.shape[1])
        if(input_feats.is_cuda):            
            sample_features = sample_features.cuda()
        sampling_grid = torch.floor(sampling_grid)
        for idx in range(batch):
            sample_features[idx,:,:] = input_feats[idx,:,sampling_grid[idx,:,0].type(torch.LongTensor),sampling_grid[idx,:,1].type(torch.LongTensor),sampling_grid[idx,:,2].type(torch.LongTensor)].clone()    
        return sample_features

def find_peaks(param, img):
    """
    Given a (grayscale) image, find local maxima whose value is above a given
    threshold (param['thre1'])
    :param img: Input image (2d array) where we want to find peaks
    :return: 2d np.array containing the [x,y] coordinates of each peak found
    in the image
    """

    peaks_binary = (maximum_filter(img, footprint=generate_binary_structure(
        2, 1)) == img) * (img > param['thre1'])
    # Note reverse ([::-1]): we return [[x y], [x y]...] instead of [[y x], [y
    # x]...]
    return np.array(np.nonzero(peaks_binary)[::-1]).T


def compute_resized_coords(coords, resizeFactor = 1.):
    """
    Given the index/coordinates of a cell in some input array (e.g. image),
    provides the new coordinates if that array was resized by making it
    resizeFactor times bigger.
    E.g.: image of size 3x3 is resized to 6x6 (resizeFactor=2), we'd like to
    know the new coordinates of cell [1,2] -> Function would return [2.5,4.5]
    :param coords: Coordinates (indices) of a cell in some input array
    :param resizeFactor: Resize coefficient = shape_dest/shape_source. E.g.:
    resizeFactor=2 means the destination array is twice as big as the
    original one
    :return: Coordinates in an array of size
    shape_dest=resizeFactor*shape_source, expressing the array indices of the
    closest point to 'coords' if an image of size shape_source was resized to
    shape_dest
    """

    # 1) Add 0.5 to coords to get coordinates of center of the pixel (e.g.
    # index [0,0] represents the pixel at location [0.5,0.5])
    # 2) Transform those coordinates to shape_dest, by multiplying by resizeFactor
    # 3) That number represents the location of the pixel center in the new array,
    # so subtract 0.5 to get coordinates of the array index/indices (revert
    # step 1)
    return (np.array(coords, dtype=float) + 0.5) * resizeFactor - 0.5


def NMS(heatmaps, upsampFactor=1., param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}, bool_refine_center=True, bool_gaussian_filt=False,NUM_JOINTS = 1):
    """
    NonMaximaSuppression: find peaks (local maxima) in a set of grayscale images
    :param heatmaps: set of grayscale images on which to find local maxima (3d np.array,
    with dimensions image_height x image_width x num_heatmaps)
    :param upsampFactor: Size ratio between CPM heatmap output and the input image size.
    Eg: upsampFactor=16 if original image was 480x640 and heatmaps are 30x40xN
    :param bool_refine_center: Flag indicating whether:
     - False: Simply return the low-res peak found upscaled by upsampFactor (subject to grid-snap)
     - True: (Recommended, very accurate) Upsample a small patch around each low-res peak and
     fine-tune the location of the peak at the resolution of the original input image
    :param bool_gaussian_filt: Flag indicating whether to apply a 1d-GaussianFilter (smoothing)
    to each upsampled patch before fine-tuning the location of each peak.
    :return: a NUM_JOINTS x 4 np.array where each row represents a joint type (0=nose, 1=neck...)
    and the columns indicate the {x,y} position, the score (probability) and a unique id (counter)
    """
    # MODIFIED BY CARLOS: Instead of upsampling the heatmaps to heatmap_avg and
    # then performing NMS to find peaks, this step can be sped up by ~25-50x by:
    # (9-10ms [with GaussFilt] or 5-6ms [without GaussFilt] vs 250-280ms on RoG
    # 1. Perform NMS at (low-res) CPM's output resolution
    # 1.1. Find peaks using scipy.ndimage.filters.maximum_filter
    # 2. Once a peak is found, take a patch of 5x5 centered around the peak, upsample it, and
    # fine-tune the position of the actual maximum.
    #  '-> That's equivalent to having found the peak on heatmap_avg, but much faster because we only
    #      upsample and scan the 5x5 patch instead of the full (e.g.) 480x640
      

    # For every peak found, win_size specifies how many pixels in each
    # direction from the peak we take to obtain the patch that will be
    # upsampled. Eg: win_size=1 -> patch is 3x3; win_size=2 -> 5x5
    # (for BICUBIC interpolation to be accurate, win_size needs to be >=2!)
    win_size = 2
    heatmaps = heatmaps.cpu().data.numpy().transpose(0, 2, 3, 1)
    peaks = np.zeros((heatmaps.shape[0],NUM_JOINTS, 2))
    for batch in range(heatmaps.shape[0]):
        for joint in range(NUM_JOINTS):
            map_orig = heatmaps[batch,:, :, joint]
            peak_coords = find_peaks(param, map_orig)
            
            for i, peak in enumerate(peak_coords):
                if bool_refine_center:
                    x_min, y_min = np.maximum(0, peak - win_size)
                    x_max, y_max = np.minimum(
                        np.array(map_orig.T.shape) - 1, peak + win_size)
    
                    # Take a small patch around each peak and only upsample that
                    # tiny region
                    patch = map_orig[y_min:y_max + 1, x_min:x_max + 1]
    
                    # Gaussian filtering takes an average of 0.8ms/peak (and there might be
                    # more than one peak per joint!) -> For now, skip it (it's
                    # accurate enough)
                    patch = gaussian_filter(
                        patch, sigma=3) if bool_gaussian_filt else patch
    
                    # Obtain the coordinates of the maximum value in the patch
                    location_of_max = np.unravel_index(
                        patch.argmax(), patch.shape)
                    # Remember that peaks indicates [x,y] -> need to reverse it for
                    # [y,x]
                    location_of_patch_center = np.array(peak[::-1] - [y_min, x_min], dtype=float)
                    # Calculate the offset wrt to the patch center where the actual
                    # maximum is
                    refined_center = (location_of_max - location_of_patch_center)
                    peak_score = patch[location_of_max] #use to debug
                else:
                    refined_center = [0, 0]
                    # Flip peak coordinates since they are [x,y] instead of [y,x]
                    peak_score = map_orig[tuple(peak[::-1])] #use to debug
                peaks[batch,joint, :] = tuple([int(round(x)) for x in np.array(peak_coords[i], dtype=float) + refined_center[::-1]])    
                peaks[batch,joint, :] = compute_resized_coords(peaks[batch,joint, :], upsampFactor)
    peaks = torch.from_numpy(peaks)

    return peaks 


def build_coordnorm_volumes(batch_size,nvox): #the function will form the norm coord volume,the value range in each axis is -1~1    
    grid_range_x = torch.linspace(-1, 1, nvox)
    grid_range_y = torch.linspace(-1, 1, nvox)
    grid_range_z = torch.linspace(-1, 1, nvox)
    grid = torch.stack(torch.meshgrid(grid_range_x, grid_range_y, grid_range_z), dim=-1)    
    grids = grid.repeat(batch_size,1,1,1,1)
    return grids

def integrate_tensor_3d_with_coordinates(volumes,softmax=True):
    batch_size, n_volumes, x_size, y_size, z_size = volumes.shape
    coord_volumes = build_coordnorm_volumes(batch_size,x_size)
    if(volumes.is_cuda):
        coord_volumes = coord_volumes.cuda()
        
    volumes = volumes.reshape((batch_size, n_volumes, -1))
    if softmax:
        volumes = nn.functional.softmax(volumes, dim=2)
    else:
        volumes = nn.functional.relu(volumes)

    volumes = volumes.reshape((batch_size, n_volumes, x_size, y_size, z_size))
    coordinates = torch.einsum("bnxyz, bxyzc -> bnc", volumes, coord_volumes)

    return coordinates

class volume_viewBased_fetureenforcememt(nn.Module):
    def __init__(self):
        super(volume_viewBased_fetureenforcememt, self).__init__()    
        self.relu = nn.ReLU()        
        
    def forward(self,volume_batch):    
        batch_size, n_views, channel , volume_shape = volume_batch.shape[0],volume_batch.shape[1],volume_batch.shape[2],volume_batch.shape[3]  
        #generate other view idx for each view
        other_view_idx = []
        for i in range(n_views):
            view_list = []
            for v in range(n_views):            
                if (v==i):
                   continue
                view_list.append(v)
            other_view_idx.append(view_list)     
            
        #before give the weight based multi view feature enforcememt,we compute their norm vector first 
        volume_batch_reshape = volume_batch.view(batch_size,n_views, channel,-1).clone()
        volume_batch_reshape = volume_batch_reshape + 0.0000000001	
        volume_batch_reshape_length = ((((volume_batch_reshape**2).sum(2))**0.5).view(batch_size,n_views,1,-1)) 
        volume_batch_reshape = volume_batch_reshape/volume_batch_reshape_length
        #start to enforce the each view feature        
        for view in range(n_views):
            #make the weight for other view information
            weight_otherview = (((volume_batch_reshape[:,view,:,:].view(batch_size,1,channel,-1) * 
                                           volume_batch_reshape[:,other_view_idx[view],:,:]).sum(2))                                 
                                .view(batch_size, n_views -1,1,volume_shape,volume_shape,volume_shape))
                        
            volume_batch[:,view,:,:,:,:] = volume_batch[:,view,:,:,:,:] + (weight_otherview * volume_batch[:,other_view_idx[view],:,:,:,:]).sum(1)  
        return volume_batch  

def volume_aggregation_method(volume_batch_to_aggregate,volume_aggregation_cfg):
    batch_size, n_views, channel , volume_shape = volume_batch_to_aggregate.shape[0],volume_batch_to_aggregate.shape[1],volume_batch_to_aggregate.shape[2],volume_batch_to_aggregate.shape[3]                                                  

    if(volume_aggregation_cfg == 'sum'):
        volume_batch = volume_batch_to_aggregate.sum(1)
    elif(volume_aggregation_cfg == 'avg'):
        volume_batch = volume_batch_to_aggregate.mean(1)
    elif(volume_aggregation_cfg == 'similiar_fusion'):        
        #the parameter for remodify the simliarty value
        alfa = 0.5
        #the divisor for the final avg fusion
        avg_divisor = torch.zeros(batch_size,1,volume_shape,volume_shape,volume_shape)
	#output
        volume_batch = torch.zeros(batch_size,n_views,channel,volume_shape,volume_shape,volume_shape)        
        if(volume_batch_to_aggregate.is_cuda):
            avg_divisor = avg_divisor.cuda()
            volume_batch = volume_batch.cuda()
        #generate other view idx for each view
        other_view_idx = []
        for i in range(n_views):
            view_list = []
            for v in range(n_views):            
                if (v==i):
                   continue
                view_list.append(v)
            other_view_idx.append(view_list) 
        #before give the weight based multi view feature enforcememt,we compute their norm vector first 
        volume_batch_reshape = volume_batch_to_aggregate.view(batch_size,n_views, channel,-1).clone()
        volume_batch_reshape = volume_batch_reshape + 0.0000000001	
        volume_batch_reshape_length = ((((volume_batch_reshape**2).sum(2))**0.5).view(batch_size,n_views,1,-1)) 
        volume_batch_reshape = volume_batch_reshape/volume_batch_reshape_length 
        #caculate simliarty between two view        
        for view in range(n_views):            
            #make the weight for other view information
            simliarty_otherview = (((volume_batch_reshape[:,view,:,:].view(batch_size,1,channel,-1) * 
                                           volume_batch_reshape[:,other_view_idx[view],:,:]).sum(2))                                 
                                .view(batch_size, n_views -1,1,volume_shape,volume_shape,volume_shape))
            #remodify the simliarty value: 0 ~ inf             
            simliarty_otherview = simliarty_otherview - alfa
            simliarty_otherview = simliarty_otherview.clamp(min = 0)
            simliarty_otherview = simliarty_otherview / (simliarty_otherview + 0.0000000001)
            #get weight for current view feature to deterimine it should be keep or 0
            weight_current_view = simliarty_otherview.sum(1) / (simliarty_otherview.sum(1) + 0.0000000001)
            avg_divisor[weight_current_view > 0] = avg_divisor[weight_current_view > 0]  + 1
            volume_batch[:,view,:,:,:,:] = volume_batch[:,view,:,:,:,:] + (weight_current_view * volume_batch_to_aggregate[:,view,:,:,:,:])
            
        avg_divisor[avg_divisor==0] = avg_divisor[avg_divisor==0] + 1
        volume_batch = volume_batch.sum(1) / avg_divisor
    return volume_batch
