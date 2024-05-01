#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:54:15 2024

@author: venkatesh
"""

import warnings

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import find_contours

def hausdorff_distance_mask(image0, image1, method = 'standard'):
    """Calculate the Hausdorff distance between the contours of two segmentation masks.
    Parameters
    ----------
    image0, image1 : ndarray
        Arrays where ``True`` represents a pixel from a segmented object. Both arrays must have the same shape.
    method : {'standard', 'modified'}, optional, default = 'standard'
        The method to use for calculating the Hausdorff distance.
        ``standard`` is the standard Hausdorff distance, while ``modified``
        is the modified Hausdorff distance.
    Returns
    -------
    distance : float
        The Hausdorff distance between coordinates of the segmentation mask contours in
        ``image0`` and ``image1``, using the Euclidean distance.
    Notes
    -----
    The Hausdorff distance [1]_ is the maximum distance between any point on the 
    contour of ``image0`` and its nearest point on the contour of ``image1``, and 
    vice-versa.
    The Modified Hausdorff Distance (MHD) has been shown to perform better
    than the directed Hausdorff Distance (HD) in the following work by
    Dubuisson et al. [2]_. The function calculates forward and backward
    mean distances and returns the largest of the two.
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Hausdorff_distance
    .. [2] M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
       matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
       :DOI:`10.1109/ICPR.1994.576361`
       http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.1.8155
    Examples
    --------
    >>> ground_truth = np.zeros((100, 100), dtype=bool)
    >>> predicted = ground_truth.copy()
    >>> ground_truth[30:71, 30:71] = disk(20)
    >>> predicted[25:65, 40:70] = True
    >>> hausdorff_distance_mask(ground_truth, predicted)
    11.40175425099138
    """
    
    if method not in ('standard', 'modified'):
        raise ValueError(f'unrecognized method {method}')
    
    a_points = find_contours(image0>0)
    b_points = find_contours(image1>0)
    
    # Handle empty sets properly:
    # - if both sets are empty, return zero
    # - if only one set is empty, return infinity
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    a_points = np.concatenate(a_points)
    b_points = np.concatenate(b_points)
    
    fwd, bwd = (
        cKDTree(a_points).query(b_points, k=1)[0],
        cKDTree(b_points).query(a_points, k=1)[0],
    )

    if method == 'standard':  # standard Hausdorff distance
        return max(max(fwd), max(bwd))
    elif method == 'modified':  # modified Hausdorff distance
        return max(np.mean(fwd), np.mean(bwd))
# loading the model.

def compute_loss(y_hat, y):
    return nn.BCELoss()(y_hat, y)

#%%

def binary_entropy(prediction_map):
    """
    Calculate the binary entropy of a binary prediction map.

    Parameters:
    prediction_map (torch.Tensor): Binary prediction map of shape (batch_size, 1, height, width).

    Returns:
    torch.Tensor: Binary entropy value.
    """
    # Ensure input tensor is on CPU and in float format
    prediction_map = prediction_map.cpu().float()
    
    # Calculate entropy
    entropy_value = - (prediction_map * torch.log2(prediction_map + 1e-20))
    
    return entropy_value
#%%
import pytorch_ssim
import pytorch_iou
from loss.contour_loss import ContourLoss

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def bce_ssim_loss(pred,target):

	bce_out = bce_loss(pred,target)
	ssim_out = 1 - ssim_loss(pred,target)

# 	iou_out = iou_loss(pred,target)
# 	loss = bce_out + ssim_out + iou_out

	iou_out = iou_loss(pred,target)
	log_cosh_dice_loss_value=torch.log((torch.exp(iou_out)+torch.exp(-iou_out))/2)
	loss = bce_out + ssim_out + iou_out
# 	loss = iou_out


	return loss





def muti_bce_loss_fusion_7(d0, d1, d2, d3, d4, d5, d6,d7,  labels_v):

	loss0 = bce_ssim_loss(d0,labels_v)
	loss1 = bce_ssim_loss(d1,labels_v)
	loss2 = bce_ssim_loss(d2,labels_v)
	loss3 = bce_ssim_loss(d3,labels_v)
	loss4 = bce_ssim_loss(d4,labels_v)
	loss5 = bce_ssim_loss(d5,labels_v)
	loss6 = bce_ssim_loss(d6,labels_v)
	loss7 = bce_ssim_loss(d7,labels_v)
	#ssim0 = 1 - ssim_loss(d0,labels_v)

	# iou0 = iou_loss(d0,labels_v)
	#loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7#+ 5.0*lossa
	return loss


def muti_bce_loss_fusion_8(d0, d1, d2, d3, d4, d5, d6,d7,d8,  labels_v):

	loss0 = bce_ssim_loss(d0,labels_v)
	loss1 = bce_ssim_loss(d1,labels_v)
	loss2 = bce_ssim_loss(d2,labels_v)
	loss3 = bce_ssim_loss(d3,labels_v)
	loss4 = bce_ssim_loss(d4,labels_v)
	loss5 = bce_ssim_loss(d5,labels_v)
	loss6 = bce_ssim_loss(d6,labels_v)
	loss7 = bce_ssim_loss(d7,labels_v)
	loss8 = bce_ssim_loss(d8,labels_v)

	#ssim0 = 1 - ssim_loss(d0,labels_v)

	# iou0 = iou_loss(d0,labels_v)
	#loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7+loss8#+ 5.0*lossa
	return loss



def muti_bce_loss_fusion(d0, d1,  labels_v):

	loss0 = bce_ssim_loss(d0,labels_v)
	loss1 = bce_ssim_loss(d1,labels_v)

	loss = loss0 + loss1
	return loss

def muti_bce_loss_fusion_6(d0, d1, d2, d3, d4, d5, d6,  labels_v):

	loss0 = bce_ssim_loss(d0,labels_v)
	loss1 = bce_ssim_loss(d1,labels_v)
	loss2 = bce_ssim_loss(d2,labels_v)
	loss3 = bce_ssim_loss(d3,labels_v)
	loss4 = bce_ssim_loss(d4,labels_v)
	loss5 = bce_ssim_loss(d5,labels_v)
	loss6 = bce_ssim_loss(d6,labels_v)
# 	loss7 = bce_ssim_loss(d7,labels_v)
	#ssim0 = 1 - ssim_loss(d0,labels_v)

	# iou0 = iou_loss(d0,labels_v)
	#loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 #+ loss7#+ 5.0*lossa
	return loss
#%%

from loss.contour_loss import ContourLoss

contour_loss_calculator = ContourLoss()

def bce_ssim_contour_loss(pred,target):

	lambda_val=0.1
	bce_out = bce_loss(pred,target)
	ssim_out = 1 - ssim_loss(pred,target)
	iou_out = iou_loss(pred,target)
	contour_loss_out=contour_loss_calculator(pred,target);
	log_cosh_dice_loss_value=torch.log((torch.exp(iou_out)+torch.exp(-iou_out))/2)
	loss = bce_out + ssim_out + iou_out+lambda_val*contour_loss_out


	return loss


def muti_bce_loss_with_contour_loss_fusion_7(d0, d1, d2, d3, d4, d5, d6,d7,  labels_v):

	loss0 = bce_ssim_contour_loss(d0,labels_v)
	loss1 = bce_ssim_contour_loss(d1,labels_v)
	loss2 = bce_ssim_contour_loss(d2,labels_v)
	loss3 = bce_ssim_contour_loss(d3,labels_v)
	loss4 = bce_ssim_contour_loss(d4,labels_v)
	loss5 = bce_ssim_contour_loss(d5,labels_v)
	loss6 = bce_ssim_contour_loss(d6,labels_v)
	loss7 = bce_ssim_contour_loss(d7,labels_v)
	#ssim0 = 1 - ssim_loss(d0,labels_v)

	# iou0 = iou_loss(d0,labels_v)
	#loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6+ loss7#+ 5.0*lossa
	return loss







#%%
def calculate_exp_of_margin(probability_map):
    
    class_0_prob_map=probability_map
    class_1_prob_map=1-probability_map
    # Calculate margin for each pixel
    #print('margin gap',)
    return torch.mean(torch.exp((class_0_prob_map)-(class_1_prob_map)))


def calculate_exp_of_margin_on_all_pixels(probability_map):
    
    class_0_prob_map=probability_map
    class_1_prob_map=1-probability_map
    # Calculate margin for each pixel
    #print('margin gap',)
    return torch.mean(torch.exp((class_0_prob_map)-(class_1_prob_map)))


def calculate_margin_updated(probability_map):
    
    class_0_prob_map=probability_map
    class_1_prob_map=1-probability_map
    # Calculate margin for each pixel
    #print('margin gap',)
    return torch.exp((class_0_prob_map)-(class_1_prob_map))

def calculate_positive_margin(probability_map):
    # Calculate the probability maps for class 0 and class 1
    class_0_prob_map = probability_map
    class_1_prob_map = 1 - probability_map
    
    # Calculate the margin for each pixel
    margin = class_0_prob_map - class_1_prob_map
    # print('class_0_prob_map - class_1_prob_map')
    # print(margin)
    margin[margin<0]=0

    # if(torch.count_nonzero(margin)>0):
    #     return torch.sum(margin)/torch.count_nonzero(margin)
    # else:
    #     return torch.tensor(0);
    
    return torch.mean(margin)

def calculate_negative_margin(probability_map):
    # Calculate the probability maps for class 0 and class 1
    class_0_prob_map = probability_map
    class_1_prob_map = 1 - probability_map
    
    # Calculate the margin for each pixel
    margin = class_0_prob_map - class_1_prob_map
    # print('class_0_prob_map - class_1_prob_map')
    # print(margin)
    margin[margin>0]=0

    # if(torch.count_nonzero(margin)>0):
    #     return torch.sum(margin)/torch.count_nonzero(margin)
    # else:
    #     return torch.tensor(0);
    return torch.mean(margin)
def error_in_probability_map(probability_map):
    return torch.mean(1-probability_map);
#%%
def write_mask_on_image(input_image,input_mask,expected_color=(255,0,0)):
    
    temp=input_image
    # make a edges/boundary for the mask
    edges = cv2.Canny(input_mask,0,255)
    
    
    # replacing the mask boundary in the given image
    temp[:,:,0][edges==255]=expected_color[0]
    temp[:,:,1][edges==255]=expected_color[1]
    temp[:,:,2][edges==255]=expected_color[2]
    
    #plt.imshow(img1)
    
    #cv2.imshow('img1',img1)
    
    cv2.imwrite("masked_image.png", cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))

    return cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)

def write_mask_on_image(input_image,input_mask,expected_color=(255,0,0)):
    
    temp=input_image
    # make a edges/boundary for the mask
    edges = cv2.Canny(input_mask,0,255)
    
    
    # replacing the mask boundary in the given image
    temp[:,:,0][edges==255]=expected_color[0]
    temp[:,:,1][edges==255]=expected_color[1]
    temp[:,:,2][edges==255]=expected_color[2]
    
    #plt.imshow(img1)
    
    #cv2.imshow('img1',img1)
    
    cv2.imwrite("masked_image.png", cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))

    return cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)


#%%
#plt.imshow(write_mask_on_image(input_image, input_mask,expected_color=(0,255,0)))

def cross_section_area(mask):
    temp=mask[mask==255]
    #print(temp.shape)


def another_metrics(ground_truth_mask,generated_mask):
    #print(ground_truth_mask.shape,generated_mask.shape)
    #print(np.unique(ground_truth_mask),np.unique(generated_mask))
    
    TP=np.count_nonzero(ground_truth_mask[generated_mask==255]==255)
    #print(TP)
    
    temp_ground=np.copy(ground_truth_mask)
    temp_ground[generated_mask==255]=0
    FN=np.count_nonzero(temp_ground)

    temp_generated=np.copy(generated_mask)
    temp_generated[ground_truth_mask==255]=0
    FP=np.count_nonzero(temp_generated)
    
    TN=464*352-(TP+FP+FN)
    
    #print(TP,TN,FP,FN)
    
    if(TP!=0):
        accuray=(TP+TN)/(TP+FP+FN+TN)
        precision=(TP)/(TP+FP)
        recall=(TP)/(TP+FN)
        F1score=(2*TP)/(2*TP+FP+FN)
        Threatscore=(TP)/(TP+FN+FP)
        correction_effort=(FP+FN)/(TP+FN)

    else:
        precision=0
        recall=0
        accuray=(TP+TN)/(TP+FP+FN+TN)
        F1score=0
        Threatscore=0
        correction_effort=(FP+FN)/(TP+FN)
        
        

    #print([accuray,precision,recall])
    
    return accuray,precision,recall,F1score,Threatscore,correction_effort
#%%


def tic():
    # Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        #print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
        print(str(time.time() - startTime_for_tictoc) )
    else:
        print("Toc: start time not set")
#%%
# IT IS FOR WRITING THE MASKS
def write_masks(output_cpu,directory,iteration,is_actual):
        img=output_cpu
        img=(img).astype(np.uint8)
        img=Image.fromarray(img)
        if(is_actual):
            img.save(directory+str(iteration)+'_mask_actual.tif' )
            #print('actual mask writing...')
        else:
            img.save(directory+str(iteration)+'_mask_generated.tif' )
            #print('generated mask writing...')

# IT IS FOR WRITING THE IMAGES.

def write_images(output_cpu,directory,iteration,is_actual):
        img=output_cpu
        img=(img).astype(np.uint8)
        img=Image.fromarray(img)
        if(is_actual):
            img.save(directory+str(iteration)+'_mask_actual.jpg' )
            #print('actual mask writing...')
        else:
            img.save(directory+str(iteration)+'_mask_generated.jpg' )
            #print('generated mask writing...')


def write_masks_appened(output_cpu,directory,iteration):
        img=output_cpu[i]
        img=(img).astype(np.uint8)
        img=Image.fromarray(img)
        img.save(directory+str(i)+'_mask_on_image.tif' )
        
        
#%%

def perturb_weights(model, std=0.00):
    for param in model.parameters():
        noise = param*std
        param.data.add_(noise)

# perturb_weights(model, std=0.06)

#%%
def add_gaussian_noise(image, intensity=0.3):
    image=image.numpy()
    std_dev=int(intensity*255)
    noise = np.random.normal(0, std_dev, image.shape)
    noisy_image = np.clip(image + noise, 0, 255);
    noisy_image=noisy_image.astype(np.uint8)
    noisy_image=torch.tensor(noisy_image)
    return noisy_image