'''from https://github.com/garyzhao/SemGCN/blob/master/common/loss.py''' 
import torch
import numpy as np
from spacepy import pycdf


def load_PredictAndLabel(predict_path = '',label_path = '',joint_number = 17 ,toTorch = True):
    #load data to numpy
    cdf_data = pycdf.CDF(predict_path)
    pred_cdf = cdf_data.copy()
    for k in pred_cdf.keys(): Pred = pred_cdf[k][:]
    cdf_data.close()
    cdf_data = pycdf.CDF(label_path)
    GTs_cdf = cdf_data.copy()
    for k in GTs_cdf.keys(): GTs = GTs_cdf[k][:]
    cdf_data.close()
    #reshape to the size: [samples,joints,axis]
    Pred = Pred.reshape((Pred.shape[0],joint_number,3))
    GTs = GTs.reshape((GTs.shape[0],joint_number,3))
    if(toTorch == True):
        Pred = torch.from_numpy(Pred)
        GTs = torch.from_numpy(GTs)
    return  Pred,GTs
    
def mpjpe(predicted, target): 
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    input shape : [samples,joints,axis]
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

def pa_mpjpe(predicted_torch, target_torch):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    input shape : [samples,joints,axis]
    """
    assert predicted_torch.shape == target_torch.shape
    
    if((predicted_torch.is_cuda) or (target_torch.is_cuda)):
        predicted = predicted_torch.cpu().numpy()
        target = target_torch.cpu().numpy()
    else:    
        predicted = predicted_torch.numpy()
        target = target_torch.numpy()    

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))

def rootr_mpjpe(predicted, target, root_idx): 
    """
    root reletead Mean per-joint position error (i.e. mean Euclidean distance),    
    input shape(predicted, target) : [samples,joints,axis]
    root_idx is the index of the root joint
    """
    assert predicted.shape == target.shape
    predict_root_reletead = predicted - predicted[:,[root_idx],:]
    target_root_reletead = target - target[:,[root_idx],:] 
    return torch.mean(torch.norm(predict_root_reletead - target_root_reletead, dim=len(target_root_reletead.shape) - 1))

def pck_3d(predicted, target):
    """
    percentage of correct joint(defined from paper: Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision)
    input shape : [samples,joints,axis]
    threshold : 150mm (difference between predicted and target joint)
    """
    difference = torch.norm(predicted - target, dim=len(target.shape) - 1)
    difference = difference.view(-1)
    
    correct_joint = (difference < 150).float()
    
    return (torch.sum(correct_joint) / correct_joint.size()[0]) * 100