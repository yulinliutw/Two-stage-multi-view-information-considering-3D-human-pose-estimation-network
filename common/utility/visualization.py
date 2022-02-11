import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
  

hm_edges = np.array([[0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13],
                    [8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6]])

def vis_2d_feature_point(feature_map,point):
    subplot_size = [3,3]
    
    color_list_css4 = list(sorted(mcolors.CSS4_COLORS.keys()))
    color_list = []
    for i in range(feature_map.size()[0]):
        color_list.append(color_list_css4[i*4])   
    
    plt.figure()
    idx = 0
    for y in np.linspace(-1,1,3):
        for x in np.linspace(-1,1,3): 
            idx = idx + 1
            feature_vector = feature_map[:,point[1].astype('int32')+y.astype('int32'),point[0].astype('int32')+x.astype('int32')].cpu().numpy()
            plt.subplot(subplot_size[0],subplot_size[1],idx)
            plt.bar(np.arange(feature_map.size()[0]),feature_vector,color = color_list)
            plt.xticks(np.arange(feature_map.size()[0]),size='small')
            plt.yticks(np.linspace(feature_vector.min(),feature_vector.max(),10))     
            
    plt.show() 
    
def vis_3d_feature_point(feature_map,point):   
    subplot_size = [3,3]
    shift = [[0,0,0],[0,0,1],[0,0,-1],[-1,0,0],[1,0,0],[0,-1,0],[0,1,0]]
    color_list_css4 = list(sorted(mcolors.CSS4_COLORS.keys()))
    color_list = []
    for i in range(feature_map.size()[0]):
        color_list.append(color_list_css4[i*4])  
        
    plt.figure()
    for idx in range(7):
        plt.subplot(subplot_size[0],subplot_size[1],idx+1) 
        feature_vector = feature_map[:,point[0].astype('int32')+shift[idx][0],point[1].astype('int32')+shift[idx][1],point[2].astype('int32')+shift[idx][2]].cpu().numpy()
        plt.subplot(subplot_size[0],subplot_size[1],idx+1)
        plt.bar(np.arange(feature_map.size()[0]),feature_vector,color = color_list)
        plt.xticks(np.arange(feature_map.size()[0]),size='small')
        plt.yticks(np.linspace(feature_vector.min(),feature_vector.max(),10)) 
    plt.show()
    
def vis_3D_array_By2D(input_tensor,indicate_vis_channel = 0,is_heatmap = True,GT_location = np.zeros(3)):
    if(input_tensor.is_cuda):
        input_tensor_np = input_tensor.cpu().numpy()
    else:
        input_tensor_np = input_tensor.numpy()        
    #the array to take the value from input_tensor_np per channel
    volume_temp = input_tensor_np[indicate_vis_channel,:,:,:]
    
    #visualize the result by cutting the 3d array to several 2d figure along the z axis
    subplot_size = [4,8] #define the subplot size to visualize the 3D array per z dim
    if(volume_temp.shape[2] == 32):
        subplot_size = [4,8]
    
    fig = plt.figure()
    if(is_heatmap): #show where is the ground truth location in current space
        fig.suptitle('The ground truth was locate in:('+str(GT_location[0])+','+str(GT_location[1])+','+str(GT_location[2])+')')
    for idx in range(volume_temp.shape[2]):    
        plt.subplot(subplot_size[0],subplot_size[1],idx+1)
        plt.title('z ='+str(idx),fontsize=8,color ='red')
        volume_temp_2D = volume_temp[:,:,idx]
        im = plt.imshow(volume_temp_2D,vmin=volume_temp.min(), vmax=volume_temp.max())
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    plt.colorbar(im,cax=cbar_ax)    
    plt.show()
    
def vis_2D_array(input_tensor):
    if(input_tensor.is_cuda):
        input_tensor_np = input_tensor.cpu().numpy()
    else:
        input_tensor_np = input_tensor.numpy()    
    #visualize the result by cutting the 3d array to several 2d figure along the z axis
    subplot_size = [4,8] #define the subplot size to visualize the 3D array per z dim
    if(input_tensor_np.shape[0] == 32):
        subplot_size = [4,8]
    if(input_tensor_np.shape[0] == 128):
        subplot_size = [8,16]
    fig = plt.figure()
    for idx in range(input_tensor_np.shape[0]):    
        plt.subplot(subplot_size[0],subplot_size[1],idx+1)
        plt.title('c ='+str(idx),fontsize=8,color ='red')        
        im = plt.imshow(input_tensor_np[idx,:,:],vmin = input_tensor_np.min(), vmax = input_tensor_np.max())
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    plt.colorbar(im,cax=cbar_ax)    
    plt.show()    
    

def show_2d(img, points, edges = None):
    num_joints = points.shape[0]
    points = ((points.reshape(num_joints, -1))).astype(np.int32)
    for j in range(num_joints):
        cv2.circle(img, (points[j, 0], points[j, 1]), 3, (255, 0, 0), -1, -1)
    if edges != None:  
        for e in edges:
            if points[e].min() > 0:
                cv2.line(img, (points[e[0], 0], points[e[0], 1]),
                        (points[e[1], 0], points[e[1], 1]), c, 2)
    return img

def get_joint_prediction_origin(input_pred,vmax,vmin):
    mean_label = ((vmax)+(vmin))/2
    scale_label = ((vmax)-(vmin))/2
    
    input_pred[:,:,0] = (input_pred[:,:,0] * scale_label[:,0].view(-1,1)) + mean_label[:,0].view(-1,1) 
    input_pred[:,:,1] = (input_pred[:,:,1] * scale_label[:,1].view(-1,1)) + mean_label[:,1].view(-1,1) 
    input_pred[:,:,2] = (input_pred[:,:,2] * scale_label[:,2].view(-1,1)) + mean_label[:,2].view(-1,1)  
    return input_pred

def show_3d(points, c='b', marker='o', edges=None):
    if edges == 'hm_edges':
      edges = hm_edges
    fig = plt.figure()
    ax = fig.add_subplot((111),projection='3d')
    ax.grid(False)
    oo = 1e10
    xmax, ymax, zmax = -oo, -oo, -oo
    xmin, ymin, zmin = oo, oo, oo
    
    points = points.reshape(-1, 3)
    x, y, z = np.zeros((3, points.shape[0]))
    for j in range(points.shape[0]):
      x[j] = points[j, 0].copy()
      y[j] = points[j, 1].copy()
      z[j] = points[j, 2].copy()
      xmax = max(x[j], xmax)
      ymax = max(y[j], ymax)
      zmax = max(z[j], zmax)
      xmin = min(x[j], xmin)
      ymin = min(y[j], ymin)
      zmin = min(z[j], zmin)
    if c == 'auto':
      c = [(z[j] + 0.5, y[j] + 0.5, x[j] + 0.5) for j in range(points.shape[0])]
    ax.scatter(x, y, z, s = 100, c = c, marker = marker)    
    if edges != None:
        for e in edges:
          ax.plot(x[e], y[e], z[e], c='r')

     
    max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)
    for xb, yb, zb in zip(Xb, Yb, Zb):
      ax.plot([xb], [yb], [zb], 'w')
    
    plt.show()  
    
def plot_LearningCurve(train_loss, valid_loss, log_path, jobName):
    '''
    Use matplotlib to plot learning curve at the end of training
    train_loss & valid_loss must be 'list' type
    '''
    plt.figure(figsize=(12,5))
    plt.title("Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    epochs = np.arange(len(train_loss))
    plt.plot(epochs, np.array(train_loss), 'r', label='train')
    plt.plot(epochs, np.array(valid_loss), 'b', label='valid')
    plt.legend()  
    plt.grid()  
    plt.savefig(os.path.join(log_path, jobName + '.png'))

def plt_show_joints(img, pts, pts_vis=None, color='ro'):
    # imshow(img)
    fig = plt.figure()
    plt.imshow(img)
    if pts_vis == None:
        pts_vis = np.ones(pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        if pts_vis[i, 0] > 0:
            plt.plot(pts[i, 0], pts[i, 1], color)
    # plt.axis('off')

def cv_draw_joints(im, kpt, vis, flip_pair_ids, color_left=(255, 0, 0), color_right=(0, 255, 0), radius=2):
    for ipt in range(0, kpt.shape[0]):
        if vis[ipt, 0]:
            cv2.circle(im, (int(kpt[ipt, 0] + 0.5), int(kpt[ipt, 1] + 0.5)), radius, color_left, -1)
    for i in range(0, flip_pair_ids.shape[0]):
        id = flip_pair_ids[i][0]
        if vis[id, 0]:
            cv2.circle(im, (int(kpt[id, 0] + 0.5), int(kpt[id, 1] + 0.5)), radius, color_right, -1)

def plot_3d_skeleton(ax, kpt_3d, kpt_3d_vis, parent_ids, flip_pair_ids, title, patch_width, patch_height, c0='r',c1='b',c2='g'):
    x_r = np.array([0, patch_width], dtype=np.float32)
    y_r = np.array([0, patch_height], dtype=np.float32)
    z_r = np.array([-patch_width / 2.0, patch_width / 2.0], dtype=np.float32)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    # joints
    X = kpt_3d[:, 0]
    Y = kpt_3d[:, 1]
    Z = kpt_3d[:, 2]
    vis_X = kpt_3d_vis[:, 0]
    #
    for i in range(0, kpt_3d.shape[0]):
        if vis_X[i]:
            ax.scatter(X[i], Z[i], -Y[i], c=c0, marker='o')
        x = np.array([X[i], X[parent_ids[i]]], dtype=np.float32)
        y = np.array([Y[i], Y[parent_ids[i]]], dtype=np.float32)
        z = np.array([Z[i], Z[parent_ids[i]]], dtype=np.float32)

        if vis_X[i] and vis_X[parent_ids[i]]:
            c = c1 # 'b'
            for j in range(0, flip_pair_ids.shape[0]):
                if i == flip_pair_ids[j][0]:
                    c = c2 # 'g'
                    break
            ax.plot(x, z, -y, c=c)
    ax.plot(x_r, z_r, -y_r, c='y')
    # ax.plot(np.array([np.min(X), np.max(X)]), np.array([np.min(Z), np.max(Z)]), -np.array([np.min(Y), np.max(Y)]), c='y')
    ax.set_title(title)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()

def debug_vis(img_path, bbox=list(), pose=list()):
    cv_img_patch_show = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    if len(bbox) > 0:
        c_x, c_y, width, height = bbox
        pt1 = (int(c_x - 1.0 * width / 2), int(c_y - 1.0 * height / 2))
        pt2 = (int(c_x + 1.0 * width / 2), int(c_y + 1.0 * height / 2))
        cv2.rectangle(cv_img_patch_show, pt1, pt2, (0, 128, 255), 3)

    #TODO: add flip pairs
    if len(pose) > 0:
        jts_3d, jts_3d_vis = pose
        for pt, pt_vis in zip(jts_3d, jts_3d_vis):
            if pt_vis[0] > 0:
                cv2.circle(cv_img_patch_show, (int(pt[0]), int(pt[1])), 3, (0,255,0), -1)

    cv2.imshow('debug visualization', cv_img_patch_show)
    cv2.waitKey(0)

def vis_compare_3d_pose(pose_a, pose_b):
    buff_large_1 = np.zeros((32, 3))
    buff_large_2 = np.zeros((32, 3))
    buff_large_1[(0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27), :] = pose_a[:-1]
    buff_large_2[(0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27), :] = pose_b[:-1]

    pose3D_1 = buff_large_1.transpose()
    pose3D_2 = buff_large_2.transpose()

    kin = np.array(
        [[0, 12], [12, 13], [13, 14], [15, 14], [13, 17], [17, 18], [18, 19], [13, 25], [25, 26], [26, 27], [0, 1],
         [1, 2],
         [2, 3], [0, 6], [6, 7], [7, 8]])

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure(1, figsize=(10, 6))
    ax = fig.add_subplot(121, projection='3d')
    ax.view_init(azim=-90, elev=15)

    for link in kin:
        ax.plot(pose3D_1[0, link], pose3D_1[2, link], -pose3D_1[1, link],
                linestyle='--', marker='o', color='green', linewidth=3.0)
        ax.plot(pose3D_2[0, link], pose3D_2[2, link], -pose3D_2[1, link],
                linestyle='-', marker=',', color='red', linewidth=3.0)
    ax.legend()

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')

    X = pose3D_1[0, :]
    Y = pose3D_1[2, :]
    Z = -pose3D_1[1, :]
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0

    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax_2d = fig.add_subplot(122)
    for link in kin:
        ax_2d.plot(pose3D_1[0, link], -pose3D_1[1, link],
                   linestyle='--', marker='o', color='green', linewidth=3.0)
        ax_2d.plot(pose3D_2[0, link], -pose3D_2[1, link],
                   linestyle='-', marker=',', color='red', linewidth=3.0)
    ax_2d.set_xlabel('X')
    ax_2d.set_ylabel('Y')
    plt.show()    
