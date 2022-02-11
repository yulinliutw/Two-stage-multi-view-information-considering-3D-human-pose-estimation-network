import sys
import os
sys.path.append(os.path.abspath('..'))
from common.eval_metrics import *
import numpy as np
import torch
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Evaluation setting')
parser.add_argument('--data_path',
                    default = '', 
                    help    = 'Path to folder of dataset')
parser.add_argument('--result_path',
                    default = '', 
                    help    = 'Path to folder of result')
parser.add_argument('--save_path',
                    default = '', 
                    help    = 'Path to folder for saving eval result')

global args
args = parser.parse_args()
action_name = ['acting','freestyle','rom','walking']
MPJPE_avg = 0
MPJPE_perAction ={}

PA_MPJPE_avg = 0
PA_MPJPE_perAction ={}

R_MPJPE_avg = 0
R_MPJPE_perAction ={}

PCK_avg = 0
PCK_perAction ={}
idx = 0
print('Start to evualate the MPJPE per action')
for a in [1,2,4]:
    #load the result and cover to tensor
    Pred,GTs = load_PredictAndLabel(predict_path = os.path.join(args.result_path,'A'+str(a)+'_pred.cdf'),label_path = os.path.join(args.result_path,'A'+str(a)+'_label.cdf'))
    if(torch.cuda.is_available()):
        Pred = Pred.cuda()
        GTs = GTs.cuda()
        
    #start to eval  
    #eval the mpjpe
    MPJPE = mpjpe(Pred, GTs)    
    MPJPE_avg = MPJPE_avg + MPJPE
    MPJPE_perAction[action_name[a-1]] =  [MPJPE.cpu().numpy()]    
    
    print('Evualate action :'+ action_name[a-1] + 'is done')
    
        
MPJPE_avg = MPJPE_avg / (a+1)    
MPJPE_perAction['avg'] =  [MPJPE_avg.cpu().numpy()]
result = pd.DataFrame(MPJPE_perAction)
result.to_csv(os.path.join(args.save_path,'MPJPE_perAction.csv'))
