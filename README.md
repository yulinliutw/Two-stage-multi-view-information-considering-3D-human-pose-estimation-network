# Two-stage-multi-view-information-considering-3D-human-pose-estimation-network
---
*Code will update later*
We propose a 3D human pose estimation network that *considers the relevance of multi-view information*. By considering the homography, we transform the extracted 2D feature into 3D space. And we explore the *relevance of image feature between different camera views*, so that each view can use other view's information to complement itself, and fuse them to describe the one 3D space. Finally, we propose a two-stage 3D human pose estimation model, to gradually adjust the prediction results. In order to show the effectiveness, we conducted evaluation tests on two widely used data sets. 
Keywords—3D human pose estimation, Multi-View Stereo, 3D feature extraction, neural networks, deep learning
It is authored by **YU LIN LIU**.

### Table of Contents
- <a href='#model-architecture'>Model Architecture</a>
- <a href='#installation'>Installation</a>
- <a href='#dataset'>Dataset</a>
- <a href='#training'>Training</a>
- <a href='#evaluation'>Evaluation</a>
- <a href='#performance'>Performance</a>

## Model Architecture
---
The network architecture can be divided into four parts: 2D image feature extraction, 2D to3D high-level feature conversion, and multi-view image information complementary and fusion algorithm and two-stage 3D human pose estimation network architecture.
**For more detail of the model, you can check the [paper](https://drive.google.com/drive/folders/1fo74BYGvcfeGEgjlLidy4y-t7nNQqp_V?usp=sharing) or the Pose_Net_train.py**

## Installation
---
- Install the [PyTorch](http://pytorch.org/) by selecting your environment on the website.
- Clone this repository.
- After excute on the python, you will found some lib need to be install, just install it.

## Dataset
---
#### Human3.6M
In my setting, I write the *h36.py* which in folder model/dataloader it can load the data for training and evaluation, just check the *h36.py*, *train.py* and *demo.py*, it can help you to know how to use it.  
**For the detail of this dataset, you can check this** [link](http://vision.imar.ro/human3.6m/description.php).

#### TotalCapture
In my setting, I write the *tc.py* which in folder model/dataloader it can load the data for training and evaluation, just check the *tc.py*, *train.py* and *demo.py*, it can help you to know how to use it.  
**For the detail of this dataset, you can check this** [link](https://cvssp.org/data/totalcapture/).

## Training
---
- Open the train.py and check the argparse setting to understand the training parameters --> config_pytorch.py which in folder model.
- Can make the personal config file .yaml for training parameter setting.
	* Note: we provide the servial pretrain weight in ./pre_train, you can load it by write the personal config file then load the setting by parse_args: --cfg, check the detail             in config_pytorch.py -->parse_args(), the example of config file can look ./parameter_setting/default.yaml* .
- Start the training.
```Shell
python train.py
```	
* Note: the folder fomat,data name fomat should follow -> [link] [link](update later)* 

## Evaluation
---
Evaluate the model on two dataset, get the testing data and run the code to saving the prediction result.
- for Human3.6M, run saving_result_h36train.py
- for TotalCapture, run saving_result_tc.py
* Note: before run the saving code, should load the setting(like model weight etc.), just check the ./parameter_setting/saving_result.yaml*
- take the saving prediction result and run the evaluation code in ./eval_methods --> check the argparse setting to give something important(like path of prediction result and   
  eval result).
  - for Human3.6M, run eval.py
  - for TotalCapture, run eval_tc.py

* Note: the link of testing data [link](update later)*

## Performance
---
(update later)
