# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 16:53:30 2021

@author: tuank
"""
# Library use in this project
import time
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import Preprocess_thresh as preThresh
import Preprocess_contour as preContour
import Preprocess_data as preData
import Detect_tumor as detect
#%% Load Data
start_time = time.time()
x = int(input('Choose Tumor number:'))
data_seg, data_t1, data_t2, data_flair = preData.imageData(x)
print('Check Data t1...')
check_t1 = preData.checkData(data_t1, 6)
print('Check Data t2...')
check_t2 = preData.checkData(data_t2, 4)
print('Check Data flair...')
check_flair = preData.checkData(data_flair, 5)
print('Finish Check')
#%% Data process
false_positive = 0
false_negative = 0
true_positive = 0
print('Start Process Image')
for i in range(data_t1.shape[2]):
    print(i)
    ''' Segmentation image'''
    img_seg = np.array(data_seg[:,:,i], np.uint8)
    
    
    ''' T1 Process'''
    img_t1 = np.array(data_t1[:,:,i], np.uint8)
    img_t1 = cv2.medianBlur(img_t1, 3)
    _, back_t1 = cv2.threshold(img_t1, 0, 256, cv2.THRESH_BINARY)
    mask_img_t1 = np.zeros(img_t1.shape, np.uint8)
    
    
    ''' T2 Process'''
    img_t2 = np.array(data_t2[:,:,i], np.uint8)
    img_t2 = cv2.medianBlur(img_t2, 3)
    _, back_t2 = cv2.threshold(img_t2, 0, 256, cv2.THRESH_BINARY)
    mask_img_t2 = np.zeros(img_t2.shape, np.uint8)
    
    
    
    ''' Flair Process'''
    img_flair = np.array(data_flair[:,:,i], np.uint8)
    img_flair = cv2.medianBlur(img_flair, 3)
    mask_img_flair = np.zeros(img_flair.shape, np.uint8)
    _, back_flair = cv2.threshold(img_flair, 0, 256, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    
    t2_pro, flair_pro, t1_pro, layer_1 = detect.Layer(img_t1, img_t2, img_flair,
                                                      check_t1, check_t2, check_flair)
    
    area_tumor = 0
    area_layer = 0
    for x in range(img_seg.shape[0]):
        for y in range(img_seg.shape[1]):
            if img_seg[x, y] > 0:
                area_tumor += 1
                if layer_1[x, y] == 255:
                    area_layer += 1
    if area_tumor != 0:
        print('Area_tumor: {}, Area_layer: {}, Area_layer/Area_tumor: {}'.format(area_tumor, area_layer, area_layer/area_tumor))
        if 0.5 < area_layer/area_tumor < 0.7:
            false_positive += 1
        elif area_layer/area_tumor <= 0.5:
            false_negative += 1
        elif area_layer/area_tumor >= 0.7:
            true_positive += 1
    plt.subplot(121)
    plt.title('Tumor')
    plt.imshow(img_seg)
    plt.subplot(122)
    plt.title('Result')
    plt.imshow(layer_1)
    plt.show()
dice_coefficient = 2 * true_positive/(2 * true_positive + false_positive + false_negative)
print('false_positive:{}, false_negative:{}, true_positive:{}, Dice: {}'.format(false_positive, false_negative, true_positive, dice_coefficient))
print("Time Main--- %s seconds ---" % (time.time() - start_time))


