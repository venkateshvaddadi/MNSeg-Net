#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 21:38:14 2023

@author: medimg
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:48:28 2023

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:04:53 2021

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 10:49:57 2021

@author: venkatesh
"""


from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch.optim as optim
import time
import tqdm
import os
from PIL import Image
import cv2

#%%
#%%

from loss.diceloss import *
#%%
from models.ResUnet import *
from models.WideResNet import *
from models.unet_model_updated import UNet_updated
from models.u2_net_model.u2net import U2NET,U2NETP
from models.basnet_model.BASNet import BASNet
#%%
#%%
from util import *
#%%

#%%
# LOADING THE MODEL 
epoch_no=12

# LOADING THE MODEL 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=epoch_no, help="enter epoch you want to test")
args = parser.parse_args()

epoch_no=args.epoch
print('\n','#'*100,'\n')
print(epoch_no)
print('\n','#'*100,'\n')

gpu_id=0


#%%
model_name='mn_net_proposed'

# loading the model.

from models.mn_net_model_proposed.mn_net_model_proposed import MN_Net_proposed

no_channels_dealing_in_and_out=128
no_channels_dealing_in_the_middle=16
model = MN_Net_proposed(in_ch=3,
                        out_ch=1,
                        no_channels_dealing_in_and_out=no_channels_dealing_in_and_out,
                        no_channels_dealing_in_the_middle=no_channels_dealing_in_the_middle)
epoch_no= 4#10 11 13 12 14 
experiment_name='mn_net_best_model'
PATH_for_experiments='saved_models/mn_net_proposed/'
PATH_for_experiment=PATH_for_experiments+"/"+experiment_name
print('proposed mn_net model loaded')


#%%

# loading the model weights and set the model in eval mode...

model_path=PATH_for_experiments+"/"+experiment_name+"/mn_net_best_model.pth"
model = torch.nn.DataParallel(model, device_ids=[gpu_id])  
model.load_state_dict(torch.load(model_path))
model=model.cuda(gpu_id)
model.eval()
#model = model.to(torch.float16)
model=model.float()
#%%

torch.save(model.state_dict(), 'model.pth')
#%%
diceloss=DiceLoss()

#%%
# data loader for the loading the dataset
from CTS_dataset import mydataloader

# LOADING THE TEST DATA.

data_path='../../data_making/aster_updated_data_nov_09_2022_with_flip/'

t_loader = mydataloader(data_path, 
                        '../../data_making/aster_updated_data_nov_09_2022_with_flip/csv_files/full_data_csv/patients_list_90_99.csv', 
                        '../../data_making/aster_updated_data_nov_09_2022_with_flip/csv_files/full_data_csv/data_with_v_h_90_99.csv')


# t_loader =  mydataloader(data_path, '../data_making/aster_updated_data_nov_09_2022_with_flip//csv_files/patients_list_100.csv',
#                           'csv_files/data_test_1_90_99_99.csv')

test_loader = DataLoader(t_loader, batch_size = 1, shuffle=False, num_workers=1)
no_test_batches=len(test_loader)
print('no_test_batches',no_test_batches)

#%%
# saving the output files.
from datetime import datetime
import os
#making directory for sving the results
print ('*******************************************************')
directory=PATH_for_experiment+'/results_'+str(epoch_no)+"/"
print('Model will be saved to  :', directory)

try:
    os.makedirs(directory)
    os.makedirs(directory+'/both_mask_on_image')

except:
    print("results are existed...")

#%%
calibration=0.004328254
testing_loss=0

accuray_list=[]
actual_cross_section_area_list=[]
computed_cross_section_area_list=[]
hausdorff_distance_list=[]
patient_video_needed=91
img_array = []

with torch.no_grad():
    for i, data in tqdm.tqdm(enumerate(test_loader)): 
            raw_image_file,mask_file,patient_id,image_no=data
            # raw_image_file=add_gaussian_noise(raw_image_file)
            image_file=raw_image_file/255
            # image_file = image_file.to(torch.float16)
            image_file=image_file.float()
            image_file=image_file.cuda(gpu_id)

            #mask_file=mask_file.float()
            # mask_file = mask_file.to(torch.float16)
            mask_file=mask_file.float()
            mask_file=mask_file.cuda(gpu_id)

            # tic()
            output, d1, d2, d3, d4, d5, d6,d7=model(image_file)
            # toc()


            # making output mask also binary
            output[output>0.5]=1
            output[output<0.5]=0

            loss = diceloss(output,mask_file/255)
            testing_loss += loss.item()

            image_file=image_file.squeeze().permute(1,2,0)
            mask_file=mask_file.squeeze()

            # making mask_file array into binary
            mask_file=mask_file/255

            image_file_generated=torch.zeros(size=(448,320,3))
            image_file_generated[:,:,0]=image_file[:,:,0]*output
            image_file_generated[:,:,1]=image_file[:,:,1]*output
            image_file_generated[:,:,2]=image_file[:,:,2]*output

            image_file_cpu=image_file.cpu().squeeze().detach().numpy()*255
            image_file_cpu=(image_file_cpu).astype(np.uint8)

            # writing actual mask 
            mask_file=mask_file.cpu().squeeze().detach().numpy()
            mask_file=mask_file*255
            mask_file=mask_file.astype(np.uint8)

            # writing the generated mask        
            output_cpu=output.cpu().squeeze().detach().numpy()
            output_cpu=output_cpu*255
            output_cpu=output_cpu.astype(np.uint8)

            temp=np.copy(image_file_cpu)

            contours_gt, _ = cv2.findContours(mask_file, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_on_image = cv2.drawContours(image_file_cpu.copy(), contours_gt, -1, (0, 255, 0), 2)
        
            # drawing predicted mask contour on given image
            contours_pred, _ = cv2.findContours(output_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_on_image = cv2.drawContours(mask_on_image.copy(), contours_pred, -1, (0, 0, 255), 2)


            accuray,precision,recall,F1score,Threatscore,correction_effort=another_metrics(mask_file, output_cpu)
            
            
            
            cs_area_from_prediction=np.count_nonzero(output_cpu==255)*calibration
            cs_area_from_actual=np.count_nonzero(mask_file==255)*calibration
            #print('CS_AREA',cs_area_from_actual,cs_area_from_prediction)
            actual_cross_section_area_list.append(cs_area_from_actual)
            computed_cross_section_area_list.append(cs_area_from_prediction)
            cv2.putText(img=mask_on_image, text=str(str(round(cs_area_from_prediction,4))), org=(275, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0,0,255),thickness=1)
            cv2.putText(img=mask_on_image, text=str(str(round(cs_area_from_actual,4))), org=(275, 70), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0,255,0),thickness=1)
            cv2.imwrite(directory+'/both_mask_on_image/'+str(patient_id.item())+'_'+str(image_no.item())+'_both_mask_on_image_'+str(round(F1score,4))+'_'+str(round(cs_area_from_actual,4))+'_'+str(round(cs_area_from_prediction,4))+'.png',mask_on_image)

            accuray_list.append([patient_id.item(),image_no.item(),1-loss.item(),accuray,precision,recall,F1score,Threatscore,correction_effort])
            hausdorff_distance_list.append(hausdorff_distance_mask(mask_file/255,output_cpu/255))
            # print('\n',patient_id.item(),image_no.item(),hausdorff_distance_mask(mask_file/255,output_cpu/255))
            
print('testing is completed.')
    

# print('dice match:',1-testing_loss/no_test_batches)


# print('dice match:',1-testing_loss/no_test_batches)
#%%
try:
    csv_results_path='csv_results/'
    os.makedirs(csv_results_path)

except:
    print("results are existed...")
#%%
temp=hausdorff_distance_list.copy()
if(np.inf in temp):
    # temp.remove(np.inf)
    temp = [x for x in temp if not np.isinf(x)]

temp=np.array(temp)
average_hus_distance=np.mean(temp)
std_hus_distance=np.std(temp)
accuray_list=np.array(accuray_list)
hausdorff_distance_list=np.array(hausdorff_distance_list)
#%%
import pandas as pd
avg_dice_score=accuray_list[:,2].mean()
accuray=accuray_list[:,3].mean()
precision=accuray_list[:,4].mean()
recall=accuray_list[:,5].mean()
F1score=accuray_list[:,6].mean()
Threatscore=accuray_list[:,7].mean()
correction_effort=accuray_list[:,8].mean()

# making the results in .csv format
dic={'patient_id':accuray_list[:,0],'image_no':accuray_list[:,1],'individual':accuray_list[:,2],'accuracy':accuray_list[:,3],'precision':accuray_list[:,4],'recall':accuray_list[:,5],'F1score':accuray_list[:,6],'Threatscore':accuray_list[:,7],'correction_effort':accuray_list[:,8],'cs_atual':actual_cross_section_area_list,'cs_computed':computed_cross_section_area_list,'hd':hausdorff_distance_list}
df = pd.DataFrame.from_dict(dic) 
df.to_csv (r''+csv_results_path+str(epoch_no)+'_'+str(round(avg_dice_score,4))+'_'+str(round(accuray,4))+'_'+str(round(precision,4))+'_'+str(round(recall,4))+"_"+str(round(average_hus_distance,4))+'.csv', index = False, header=True)

#%%
print(avg_dice_score,accuray,precision,recall,F1score,Threatscore,correction_effort)

print('#'*50)

print('avg_dice_score:',avg_dice_score)
print('precision:',precision)
print('recall:',recall)
print('hausdorff_distance:',average_hus_distance)

print('#'*100)


