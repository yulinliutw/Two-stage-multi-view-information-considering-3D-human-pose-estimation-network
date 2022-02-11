import torch
import torch.nn as nn
from torch.nn import init

def make_vgg16_block(block):
    """Builds a vgg16 block from a dictionary
    Args:
        block: a dictionary
    """
    layers = []
    for i in range(len(block)):
        one_ = block[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

def make_stages(cfg_dict):
    """Builds CPM stages from a dictionary
    Args:
        cfg_dict: a dictionary
    """
    layers = []
    for i in range(len(cfg_dict) - 1):
        one_ = cfg_dict[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    one_ = list(cfg_dict[-1].keys())
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                       kernel_size=v[2], stride=v[3], padding=v[4])
    layers += [conv2d]
    return nn.Sequential(*layers)

def Feature2D_extracter_model(outputchannel):
    """
    Creates the 2D Feature extracter model VGG16
    Returns: Module, the defined model
    """   
    block1_2 = [{'conv1_1': [3, 64, 3, 1, 1]},
              {'conv1_2': [64, 64, 3, 1, 1]},
              {'pool1_stage1': [2, 2, 0]},
              {'conv2_1': [64, 128, 3, 1, 1]},
              {'conv2_2': [128, 128, 3, 1, 1]},
              {'pool2_stage1': [2, 2, 0]}]
    
    block3 = [{'conv3_1': [128, 256, 3, 1, 1]},
              {'conv3_2': [256, 256, 3, 1, 1]},
              {'conv3_3': [256, 256, 3, 1, 1]},
              {'conv3_4': [256, 256, 3, 1, 1]}]
              
    block4 = [{'pool4_stage1': [2, 2, 0]},
              {'conv4_1': [256, 512, 3, 1, 1]},
              {'conv4_2': [512, 512, 3, 1, 1]},
              {'conv4_3_CPM': [512, 256, 3, 1, 1]},
              {'conv4_4_CPM': [256, 128, 3, 1, 1]}]

    models = {}
         
    models['block1_2'] = make_vgg16_block(block1_2)
    models['block3'] = make_vgg16_block(block3)
    models['block4'] = make_vgg16_block(block4)
    
    
    class rtpose_model(nn.Module):
        def __init__(self, model_dict,outputchannel):
            super(rtpose_model, self).__init__()
            #feature encoder
            self.model1_2 = model_dict['block1_2']
            self.model3 = model_dict['block3']
            self.model4 = model_dict['block4']            
            
            for p in self.parameters():
                p.requires_grad = False
            #extract multi-scale structure  
            self.conv_200x200 = nn.Conv2d(
                            in_channels = 128,     
                            out_channels = 32, 
                            kernel_size = 1,                           
                            stride = 1                              
                            )
            self.conv_100x100 = nn.Conv2d(
                            in_channels = 256,     
                            out_channels = 32, 
                            kernel_size = 1,                           
                            stride = 1                              
                            )
            self.conv_50x50 = nn.Sequential(
                        nn.Conv2d(
                        in_channels = 128,     
                        out_channels = 32, 
                        kernel_size = 1,                           
                        stride = 1                              
                        ),      
                        nn.Upsample(scale_factor=2, mode='bilinear')) 
            self.conv_25x25 =nn.Sequential(
                        nn.MaxPool2d(kernel_size=2, stride=2,padding=0),
                        nn.Conv2d(
                        in_channels = 128,     
                        out_channels = 32, 
                        kernel_size = 3,                           
                        stride = 1,
                        padding=1      
                        ),                                
                        nn.Upsample(scale_factor=4, mode='bilinear'))
           
            #output structure
            self.conv_concat_out = nn.Sequential(
                        nn.Conv2d(
                        in_channels = 128,     
                        out_channels = 64, 
                        kernel_size = 3,                           
                        stride = 1,
                        padding=1      
                        ),      
                        nn.BatchNorm2d(64),        
                        nn.ReLU(inplace=True))
            
            self.outputchannel = outputchannel
            self.conv_out = nn.Sequential(
                        nn.Conv2d(
                        in_channels = 64,     
                        out_channels = self.outputchannel, 
                        kernel_size = 1,                           
                        stride = 1                              
                        ),      
                        nn.BatchNorm2d(self.outputchannel),        
                        nn.ReLU(inplace=True)) 
            self._initialize_weights_norm()
        def forward(self, x):  
            #vgg16
            feature_200x200 = self.model1_2(x)
            feature_100x100 = self.model3(feature_200x200)
            feature_50x50 = self.model4(feature_100x100)
            
            #concact the multi scale feature & output final feature            
            feature_25x25 = self.conv_25x25(feature_50x50) 
            feature_50x50 = self.conv_50x50(feature_50x50)
            feature_100x100 = self.conv_100x100(feature_100x100)
            feature_200x200 = self.conv_200x200(feature_200x200)
            
            feature_cat = torch.cat([feature_25x25,feature_50x50,feature_100x100,feature_200x200],1)
            feature_cat = self.conv_concat_out(feature_cat)
            out = self.conv_out(feature_cat)
            
            return out

        def _initialize_weights_norm(self):

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.normal_(m.weight, std=0.01)
                    if m.bias is not None:  
                        init.constant_(m.bias, 0.0)

    model = rtpose_model(models,outputchannel)
    
    return model
