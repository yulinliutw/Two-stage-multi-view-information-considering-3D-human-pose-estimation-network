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
action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases'
               , 'Sitting', 'SittingDown', 'Smoking', 'Photo','Waiting', 'Walking', 'WalkDog', 'WalkTogether']
MPJPE_avg = 0
MPJPE_perAction ={}

PA_MPJPE_avg = 0
PA_MPJPE_perAction ={}

R_MPJPE_avg = 0
R_MPJPE_perAction ={}

PCK_avg = 0
PCK_perAction ={}

print('Start to evualate the MPJPE per action')
for a in range(15):
    #load the result and cover to tensor
    Pred,GTs = load_PredictAndLabel(predict_path = os.path.join(args.result_path,'A'+str(a)+'_pred.cdf'),label_path = os.path.join(args.result_path,'A'+str(a)+'_label.cdf'))
    if(torch.cuda.is_available()):
        Pred = Pred.cuda()
        GTs = GTs.cuda()
        
    #start to eval  
    #eval the mpjpe
    MPJPE = mpjpe(Pred, GTs) 
    PA_MPJPE = pa_mpjpe(Pred, GTs)
    R_MPJPE = rootr_mpjpe(Pred, GTs,6)
    PCK = pck_3d(Pred, GTs)
    
    
    MPJPE_avg = MPJPE_avg + MPJPE
    MPJPE_perAction[action_name[a]] =  [MPJPE.cpu().numpy()]
    
    PA_MPJPE_avg = PA_MPJPE_avg + PA_MPJPE
    PA_MPJPE_perAction[action_name[a]] =  [PA_MPJPE]
    
    R_MPJPE_avg = R_MPJPE_avg + R_MPJPE
    R_MPJPE_perAction[action_name[a]] =  [R_MPJPE.cpu().numpy()]
    
    PCK_avg = PCK_avg + PCK
    PCK_perAction[action_name[a]] =  [PCK.cpu().numpy()]
    
    
    
    print('Evualate action :'+ action_name[a] + 'is done')
        
MPJPE_avg = MPJPE_avg / (a+1)    
MPJPE_perAction['avg'] =  [MPJPE_avg.cpu().numpy()]
result = pd.DataFrame(MPJPE_perAction)
result.to_csv(os.path.join(args.save_path,'MPJPE_perAction.csv'))

PA_MPJPE_avg = PA_MPJPE_avg / (a+1)    
PA_MPJPE_perAction['avg'] =  [PA_MPJPE_avg]
result = pd.DataFrame(PA_MPJPE_perAction)
result.to_csv(os.path.join(args.save_path,'PA_MPJPE_perAction.csv'))

R_MPJPE_avg = R_MPJPE_avg / (a+1)    
R_MPJPE_perAction['avg'] =  [R_MPJPE_avg.cpu().numpy()]
result = pd.DataFrame(R_MPJPE_perAction)
result.to_csv(os.path.join(args.save_path,'R_MPJPE_perAction.csv'))

PCK_avg = PCK_avg / (a+1)    
PCK_perAction['avg'] =  [PCK_avg.cpu().numpy()]
result = pd.DataFrame(PCK_perAction)
result.to_csv(os.path.join(args.save_path,'PCK_perAction.csv'))