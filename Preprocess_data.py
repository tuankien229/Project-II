# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 22:35:18 2022

@author: tuank
"""

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import K_mean_function as km

def imageData(n):
    if 0 < n < 10:
        path_seg = 'D:/Spyder/Tumor_model/MICCAI_BraTS2020_TrainingData/BraTS20_Training_00'+ str(n) + '/BraTS20_Training_00' + str(n)+ '_seg.nii.gz'
        path_t1 = 'D:/Spyder/Tumor_model/MICCAI_BraTS2020_TrainingData/BraTS20_Training_00'+ str(n) + '/BraTS20_Training_00' + str(n) + '_t1.nii.gz'
        path_t2 = 'D:/Spyder/Tumor_model/MICCAI_BraTS2020_TrainingData/BraTS20_Training_00'+ str(n) + '/BraTS20_Training_00'+ str(n) + '_t2.nii.gz'
        path_flair = 'D:/Spyder/Tumor_model/MICCAI_BraTS2020_TrainingData/BraTS20_Training_00'+ str(n) + '/BraTS20_Training_00'+ str(n) + '_flair.nii.gz'
    elif 10 <= n < 100:
        path_seg = 'D:/Spyder/Tumor_model/MICCAI_BraTS2020_TrainingData/BraTS20_Training_0'+ str(n) + '/BraTS20_Training_0' + str(n)+ '_seg.nii.gz'
        path_t1 = 'D:/Spyder/Tumor_model/MICCAI_BraTS2020_TrainingData/BraTS20_Training_0'+ str(n) + '/BraTS20_Training_0'+ str(n) + '_t1.nii.gz'
        path_t2 = 'D:/Spyder/Tumor_model/MICCAI_BraTS2020_TrainingData/BraTS20_Training_0'+ str(n) + '/BraTS20_Training_0'+ str(n) + '_t2.nii.gz'
        path_flair = 'D:/Spyder/Tumor_model/MICCAI_BraTS2020_TrainingData/BraTS20_Training_0'+ str(n) + '/BraTS20_Training_0'+ str(n) + '_flair.nii.gz'
    elif 100 <= n < 1000:
        path_seg = 'D:/Spyder/Tumor_model/MICCAI_BraTS2020_TrainingData/BraTS20_Training_'+ str(n) + '/BraTS20_Training_' + str(n)+ '_seg.nii.gz'
        path_t1 = 'D:/Spyder/Tumor_model/MICCAI_BraTS2020_TrainingData/BraTS20_Training_'+ str(n) + '/BraTS20_Training_'+ str(n) + '_t1.nii.gz'
        path_t2 = 'D:/Spyder/Tumor_model/MICCAI_BraTS2020_TrainingData/BraTS20_Training_'+ str(n) + '/BraTS20_Training_'+ str(n) + '_t2.nii.gz'
        path_flair = 'D:/Spyder/Tumor_model/MICCAI_BraTS2020_TrainingData/BraTS20_Training_'+ str(n) + '/BraTS20_Training_'+ str(n) + '_flair.nii.gz'
    
    data_seg = nib.load(path_seg).get_data()
    data_t1 = nib.load(path_t1).get_data()
    data_t2 = nib.load(path_t2).get_data()
    data_flair =  nib.load(path_flair).get_data()
    
    
    return data_seg, data_t1, data_t2, data_flair
#########################################################################################################################################################

def checkData(data, K):
    count = 0
    for i in range(data.shape[2]):
        img_og = np.array(data[:,:,i], np.uint8)
        img,_ = km.K_means(img_og, K)
        count_max = 0
        count_min = 0
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if 5 < img[x,y] <= 128:
                    count_min += 1
                elif 128 < img[x,y] < 256:
                    count_max += 1
        if count_max > count_min:
            count += 1
    
    if data.shape[2] - count < count:
        return True
    return False