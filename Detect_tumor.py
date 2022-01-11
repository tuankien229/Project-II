# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 22:37:39 2022

@author: tuank
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import Preprocess_thresh as preThresh
import Preprocess_contour as preContour
import Preprocess_data as preData

def findTumor(thresh, min_size):
   mask = np.zeros(thresh.shape, np.uint8)
   output  = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
   numLabels, labels, stats, centroids = output
   for i in range(1, numLabels):
       area = stats[i, cv2.CC_STAT_AREA]
       check_area =  min_size <= area
       if check_area:
           componentMask = (labels == i).astype("uint8") * 255
           mask = cv2.bitwise_or(mask, componentMask)    
   return mask
#########################################################################################################################################################

def T1_process(img, check_t1):
    _, mask_inv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    if check_t1:
        thresh_3 = preThresh.findThresh(img, 3, 4)
        thresh_3 = cv2.bitwise_not(thresh_3, mask = mask_inv)
        
        thresh_2 = preThresh.findThresh(img, 2, 4)
        thresh_2 = cv2.bitwise_not(thresh_2, mask = mask_inv)
        
        thresh_1 = preThresh.findThresh(img, 1, 4)
        
        thresh = cv2.bitwise_and(thresh_1, thresh_3)
        thresh = cv2.bitwise_and(thresh, thresh_2)
        
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations = 1)
        thresh = findTumor(thresh, 50)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 1)
        
        img_high = np.clip(img, a_min = 106, a_max = 256).astype('uint8')
        _, thresh_high = cv2.threshold(img_high, 107, 256, cv2.THRESH_BINARY)
        thresh_high = cv2.bitwise_not(thresh_high, mask = mask_inv)
        
        thresh = cv2.bitwise_and(thresh, thresh_high)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations = 1)
        thresh = findTumor(thresh, 50)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 1)
        
        thresh = preThresh.removeThresh(thresh, mask_inv)
        
    else:
        thresh_3 = preThresh.findThresh(img, 3, 4)
        
        thresh_2 = preThresh.findThresh(img, 2, 4)
        thresh_2 = cv2.bitwise_not(thresh_2, mask = mask_inv)
        
        thresh_1 = preThresh.findThresh(img, 1, 4)
        thresh_1 = cv2.bitwise_not(thresh_1, mask = mask_inv)
        
        thresh = cv2.bitwise_and(thresh_1, thresh_3)
        thresh = cv2.bitwise_and(thresh, thresh_2)
        
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations = 1)
        thresh = findTumor(thresh, 50)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 1)
        
        img_high = np.clip(img, a_min = 106, a_max = 256).astype('uint8')
        _, thresh_high = cv2.threshold(img_high, 107, 256, cv2.THRESH_BINARY_INV)
        thresh_high = cv2.bitwise_not(thresh_high, mask = mask_inv)
        
        thresh = cv2.bitwise_and(thresh, thresh_high)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations = 1)
        thresh = findTumor(thresh, 50)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 1)
        
        thresh = preThresh.removeThresh(thresh, mask_inv)
    return thresh

#########################################################################################################################################################

def T2_process(img, check_t2, check_flair):
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    if (check_flair):
        
        thresh_3 = preThresh.findThresh(img, 3, 4)
        
        thresh_2 = preThresh.findThresh(img, 2, 4)
        thresh_2 = cv2.bitwise_not(thresh_2)
        
        thresh_1 = preThresh.findThresh(img, 1, 4)
        thresh_1 = cv2.bitwise_not(thresh_1, mask = mask)
        
        thresh = cv2.bitwise_or(thresh_1, thresh_3)
        thresh = cv2.bitwise_and(thresh, thresh_2)
        
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations = 1)
        thresh = findTumor(thresh, 50)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 1)
        
        
        
        # Split image by histogram
        
        clip_t2_max = np.clip(img, a_min = 150 , a_max = 256).astype('uint8') 
        _, t2_max = cv2.threshold(clip_t2_max, 151, 256, cv2.THRESH_BINARY)
        clip_t2_min = np.clip(img, a_min = 50, a_max = 256).astype('uint8')
        _, t2_min = cv2.threshold(clip_t2_min, 51, 256, cv2.THRESH_BINARY_INV)
        t2_min = cv2.bitwise_and(t2_min, mask)
        
        thresh_t2 = cv2.bitwise_or(t2_min, t2_max)
        thresh_t2 = cv2.morphologyEx(thresh_t2, cv2.MORPH_ERODE, kernel, iterations = 1)
        thresh_t2 = findTumor(thresh_t2, 50)
        thresh_t2 = cv2.morphologyEx(thresh_t2, cv2.MORPH_DILATE, kernel, iterations = 1)
        
        thresh = cv2.bitwise_and(thresh, thresh_t2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations = 1)
        thresh = findTumor(thresh, 50)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 1)
        
    else:
        if check_t2:
            thresh_3 = preThresh.findThresh(img, 3, 4)
            thresh_3 = cv2.morphologyEx(thresh_3, cv2.MORPH_ERODE, kernel, iterations = 1)
            thresh_3 = findTumor(thresh_3, 50)
            thresh_3 = cv2.morphologyEx(thresh_3, cv2.MORPH_DILATE, kernel, iterations = 1)
            # thresh_3 = cv2.bitwise_not(thresh_3, mask = mask)
            
            thresh_2 = preThresh.findThresh(img, 2, 4)
            thresh_2 = cv2.morphologyEx(thresh_2, cv2.MORPH_ERODE, kernel, iterations = 1)
            thresh_2 = findTumor(thresh_2, 50)
            thresh_2 = cv2.morphologyEx(thresh_2, cv2.MORPH_DILATE, kernel, iterations = 1)
            
            
            thresh_1 = preThresh.findThresh(img, 1, 4)
            # thresh_1 = cv2.bitwise_not(thresh_1, mask = mask)
            thresh_1 = cv2.morphologyEx(thresh_1, cv2.MORPH_ERODE, kernel, iterations = 1)
            thresh_1 = findTumor(thresh_1, 50)
            thresh_1 = cv2.morphologyEx(thresh_1, cv2.MORPH_DILATE, kernel, iterations = 1)
            
            clip_t2_max = np.clip(img, a_min = 150 , a_max = 256).astype('uint8') 
            _, t2_max = cv2.threshold(clip_t2_max, 151, 256, cv2.THRESH_BINARY)
            clip_t2_min = np.clip(img, a_min = 50, a_max = 256).astype('uint8')
            _, t2_min = cv2.threshold(clip_t2_min, 51, 256, cv2.THRESH_BINARY_INV)
            t2_min = cv2.bitwise_and(t2_min, mask)
            
            thresh = cv2.bitwise_and(thresh_3, t2_max)
            thresh = cv2.bitwise_or(thresh, t2_min)
            
            _, lis_cnt_thresh = preContour.drawContours(thresh)
            _, lis_cnt = preContour.drawContours(mask)
            sum_cnt = 0
            for cnt in lis_cnt:
                x, y, w, h = cnt
                sum_cnt += w*h
                
            sum_cnt_thresh = 0
            for cnt in lis_cnt_thresh:
                x, y, w, h = cnt
                sum_cnt_thresh += w*h
            if sum_cnt != 0:
                if sum_cnt_thresh/sum_cnt > 0.6:
                    thresh = cv2.bitwise_and(cv2.bitwise_not(thresh_3, mask = mask), cv2.bitwise_not(t2_max, mask = mask))
                    thresh = cv2.bitwise_or(thresh, t2_min)
            
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations = 1)
            thresh = findTumor(thresh, 50)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 1)
        else:
            thresh_3 = preThresh.findThresh(img, 3, 4)
            
            thresh_2 = preThresh.findThresh(img, 2, 4)
            thresh_2 = cv2.bitwise_not(thresh_2, mask = mask)
            
            thresh_1 = preThresh.findThresh(img, 1, 4)
            thresh_1 = cv2.bitwise_not(thresh_1, mask = mask)
            
            thresh = cv2.bitwise_or(thresh_1, thresh_3)
            thresh = cv2.bitwise_and(thresh, thresh_2)
            
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations = 1)
            thresh = findTumor(thresh, 50)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 1)
    return thresh

#########################################################################################################################################################

def Flair_process(img, check_t2, check_flair):
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    if (check_flair):
        thresh_1 = preThresh.findThresh(img, 1, 4)
        
        thresh_2 = preThresh.findThresh(img, 2, 4)
        thresh_2 = cv2.bitwise_not(thresh_2, mask = mask)
        
        thresh_3 = preThresh.findThresh(img, 3, 4)
        thresh_3 = cv2.bitwise_not(thresh_3, mask = mask)
        
        
        thresh = cv2.bitwise_or(thresh_1, thresh_3)
        thresh = cv2.bitwise_and(thresh, thresh_2)
        
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations = 1)
        thresh = findTumor(thresh, 50)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 1)
        
        # Split image by histogram
        clip_flair_min = np.clip(img, a_min = 106, a_max = 256).astype('uint8')
        _, flair_min = cv2.threshold(clip_flair_min, 107, 256, cv2.THRESH_BINARY_INV)
        clip_flair_max = np.clip(img, a_min = 150, a_max = 256).astype('uint8')
        _, flair_max = cv2.threshold(clip_flair_max, 151, 256, cv2.THRESH_BINARY_INV)
        
        thresh_flair = cv2.bitwise_and(mask, cv2.bitwise_and(flair_max, flair_min))
        thresh_flair = cv2.morphologyEx(thresh_flair, cv2.MORPH_ERODE, kernel, iterations = 1)
        thresh_flair = findTumor(thresh_flair, 50)
        thresh_flair = cv2.morphologyEx(thresh_flair, cv2.MORPH_DILATE, kernel, iterations = 1)
        
        thresh = cv2.bitwise_and(thresh_flair, thresh)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations = 1)
        thresh = findTumor(thresh, 50)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 1)
    else:
        if check_t2:
            thresh_3 = preThresh.findThresh(img, 3, 4)
            thresh_3 = cv2.morphologyEx(thresh_3, cv2.MORPH_ERODE, kernel, iterations = 1)
            thresh_3 = findTumor(thresh_3, 50)
            thresh_3 = cv2.morphologyEx(thresh_3, cv2.MORPH_DILATE, kernel, iterations = 1)
            # thresh_3 = cv2.bitwise_not(thresh_3, mask = mask)
            
            thresh_2 = preThresh.findThresh(img, 2, 4)
            thresh_2 = cv2.morphologyEx(thresh_2, cv2.MORPH_ERODE, kernel, iterations = 1)
            thresh_2 = findTumor(thresh_2, 50)
            thresh_2 = cv2.morphologyEx(thresh_2, cv2.MORPH_DILATE, kernel, iterations = 1)
            
            
            thresh_1 = preThresh.findThresh(img, 1, 4)
            thresh_1 = cv2.bitwise_not(thresh_1, mask = mask)
            thresh_1 = cv2.morphologyEx(thresh_1, cv2.MORPH_ERODE, kernel, iterations = 1)
            thresh_1 = findTumor(thresh_1, 50)
            thresh_1 = cv2.morphologyEx(thresh_1, cv2.MORPH_DILATE, kernel, iterations = 1)
            
            thresh = cv2.bitwise_or(thresh_2, thresh_3)
            thresh = cv2.bitwise_and(thresh, thresh_1)
            
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations = 1)
            thresh = findTumor(thresh, 50)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 1)
            
            _, lis_cnt_thresh = preContour.drawContours(thresh)
            _, lis_cnt = preContour.drawContours(mask)
            sum_cnt = 0
            for cnt in lis_cnt:
                x, y, w, h = cnt
                sum_cnt += w*h
                
            sum_cnt_thresh = 0
            for cnt in lis_cnt_thresh:
                x, y, w, h = cnt
                sum_cnt_thresh += w*h
            if sum_cnt != 0:
                if sum_cnt_thresh/sum_cnt > 0.6:
                    thresh = cv2.bitwise_and(thresh_3, cv2.bitwise_not(thresh_2))
                    # thresh = cv2.bitwise_not(thresh, mask = mask)
                    thresh = cv2.bitwise_and(thresh, thresh_1)
                
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations = 1)
            thresh = findTumor(thresh, 50)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 1)
            
            _, lis_cnt_thresh = preContour.drawContours(thresh)
            sum_cnt_thresh = 0
            for cnt in lis_cnt_thresh:
                x, y, w, h = cnt
                sum_cnt_thresh += w*h
            
            if sum_cnt != 0:
                if sum_cnt_thresh/sum_cnt > 0.6:
                    thresh = cv2.bitwise_and(thresh_2, thresh_1)
                    _, lis_cnt_thresh = preContour.drawContours(thresh)
                    sum_cnt_thresh = 0
                    for cnt in lis_cnt_thresh:
                        x, y, w, h = cnt
                        sum_cnt_thresh += w*h
                    if sum_cnt_thresh/sum_cnt > 0.6:
                        thresh = np.zeros(thresh.shape, np.uint8)
            
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations = 1)
            thresh = findTumor(thresh, 50)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 1)
        else:
            thresh_1 = preThresh.findThresh(img, 1, 4)
            
            thresh_2 = preThresh.findThresh(img, 2, 4)
            thresh_2 = cv2.bitwise_not(thresh_2, mask = mask)
            
            thresh_3 = preThresh.findThresh(img, 3, 4)
            
            thresh_2 = cv2.morphologyEx(thresh_2, cv2.MORPH_ERODE, kernel, iterations = 1)
            thresh_2 = findTumor(thresh_2, 50)
            thresh_2 = cv2.morphologyEx(thresh_2, cv2.MORPH_DILATE, kernel, iterations = 1)
            
            clip_flair_max = np.clip(img, a_min = 150, a_max = 256).astype('uint8')
            _, flair_max = cv2.threshold(clip_flair_max, 151, 256, cv2.THRESH_BINARY_INV)
            flair_max = cv2.bitwise_and(flair_max, mask)
            
            flair_max = cv2.morphologyEx(flair_max, cv2.MORPH_ERODE, kernel, iterations = 1)
            flair_max = findTumor(flair_max, 50)
            flair_max = cv2.morphologyEx(flair_max, cv2.MORPH_DILATE, kernel, iterations = 1)
            
            thresh = cv2.bitwise_and(thresh_2, flair_max)
            _, lis_cnt_thresh = preContour.drawContours(thresh)
            _, lis_cnt = preContour.drawContours(mask)
            sum_cnt = 0
            for cnt in lis_cnt:
                x, y, w, h = cnt
                sum_cnt += w*h
                
            sum_cnt_thresh = 0
            for cnt in lis_cnt_thresh:
                x, y, w, h = cnt
                sum_cnt_thresh += w*h
            if sum_cnt != 0 and sum_cnt_thresh/sum_cnt > 0.6:
                thresh = cv2.bitwise_and(thresh_2, cv2.bitwise_not(flair_max))
                
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations = 1)
            thresh = findTumor(thresh, 50)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 1)
    return thresh

#########################################################################################################################################################

memory_cnt = []
def Layer(img_t1, img_t2, img_flair,
          check_t1, check_t2, check_flair):
    global memory_cnt
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    _, back_t1 = cv2.threshold(img_t1, 0, 256, cv2.THRESH_BINARY)
    _, back_t2 = cv2.threshold(img_t2, 0, 256, cv2.THRESH_BINARY)
    _, back_flair = cv2.threshold(img_flair, 0, 256, cv2.THRESH_BINARY)
    '''----------------------------------------T2 Process----------------------------------------'''
    t2_pro = T2_process(img_t2, check_t2, check_flair)
    '''----------------------------------------Flair Process----------------------------------------'''
    flair_pro = Flair_process(img_flair, check_t2, check_flair)
    
    layer_1 = cv2.bitwise_or(t2_pro, flair_pro)
    layer_1 = cv2.morphologyEx(layer_1, cv2.MORPH_ERODE, kernel, iterations = 1)
    layer_1 = findTumor(layer_1, 50)
    layer_1 = cv2.morphologyEx(layer_1, cv2.MORPH_DILATE, kernel, iterations = 1)
    
    _, lis_cnt_layer_1 = preContour.drawContours(layer_1)
    t1_pro = T1_process(img_t1, check_t1)
    _, lis_cnt_t1 = preContour.drawContours(t1_pro)
        
    lis_cnt = preContour.findNewContours(back_t2)
    sum_cnt = 0
    for cnt in lis_cnt:
        x, y, w, h = cnt
        sum_cnt += w*h
    
    lis_layer_1, lis_t1 = preContour.checkContours(lis_cnt_layer_1, lis_cnt_t1)
    sum_cnt_t1 = 0
    for cnt_t1 in lis_t1:
        x, y, w, h = cnt_t1
        sum_cnt_t1 += w*h

    if len(lis_t1) != 0:
        memory_cnt.append(lis_t1)
        lis_t1, _ = preContour.checkContours(lis_cnt_t1, memory_cnt[-1])
        if len(lis_t1) != 0:
            memory_cnt.append(lis_t1)
    print('list T1: {}'.format(lis_t1))
    
    mask = np.zeros(t1_pro.shape, np.uint8)
    for cnt_t1 in lis_cnt_t1:
        X, Y, W, H = cnt_t1
        for cnt in lis_t1:
            x, y, w, h = cnt
            if 0.9 < x/X < 1.1 and 0.9 < y/Y < 1.1 and 0.9 < w/W < 1.1 and 0.9 < h/H < 1.1:
                  mask[y : y + h, x : x + w] = t1_pro[y : y + h, x : x + w]
    
    layer_1 = cv2.bitwise_or(layer_1, mask)
    layer_1 = cv2.morphologyEx(layer_1, cv2.MORPH_ERODE, kernel, iterations = 1)
    layer_1 = findTumor(layer_1, 50)
    layer_1 = cv2.morphologyEx(layer_1, cv2.MORPH_DILATE, kernel, iterations = 1)
    
    lis_cnt_layer = preContour.findNewContours(layer_1)
    sum_cnt_layer = 0
    for cnt in lis_cnt_layer:
        x, y, w, h = cnt
        sum_cnt_layer += w*h
    if sum_cnt != 0 and sum_cnt_layer/ sum_cnt > 0.8:
        layer_1 = np.zeros(layer_1.shape, np.uint8)
    
    return t2_pro, flair_pro, t1_pro, layer_1