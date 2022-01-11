# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 22:28:40 2022

@author: tuank
"""
import numpy as np
import cv2
import Preprocess_contour as preContour
import K_mean_function as km

def removeThresh(thresh, mask):
    _, lis_cnt = preContour.drawContours(mask)
    sum_cnt = 0
    for cnt in lis_cnt:
        x, y, w, h = cnt
        sum_cnt += w*h
    output  = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
    numLabels, labels, stats, centroids = output
    mask = np.zeros(thresh.shape, np.uint8)
    for i in range(1, numLabels):
       area = stats[i, cv2.CC_STAT_AREA]
       if sum_cnt != 0 and area/sum_cnt < 0.125:
           componentMask = (labels == i).astype("uint8") * 255
           mask = cv2.bitwise_or(mask, componentMask)    
    return mask
#########################################################################################################################################################

def findThresh(img, i, K):
    _, back_gr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    
    og_img = img
    _, lis_seg = km.K_means(og_img, K)
    seg = lis_seg[i]
    _, thresh_seg = cv2.threshold(seg, 0, 255, cv2.THRESH_BINARY)
    
    # Rotated image to find thresh then rotated again
    rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
    _, lis_seg_180 = km.K_means(rotated_180, K)
    seg_180 = lis_seg_180[i]
    _, thresh_180 = cv2.threshold(seg_180, 0, 255, cv2.THRESH_BINARY)
    thresh_180 = cv2.rotate(thresh_180, cv2.ROTATE_180)
    
    # Flip image to find thresh then flip agian
    flip_img = cv2.flip(img, 1)
    _, lis_seg_flip = km.K_means(flip_img, K)
    seg_flip = lis_seg_flip[i]
    _, thresh_flip = cv2.threshold(seg_flip, 0, 255, cv2.THRESH_BINARY)
    thresh_flip = cv2.flip(thresh_flip, 1)
    
    
    # Finaly stack all thresh image we found to find final thresh
    thresh = cv2.bitwise_and(thresh_seg, thresh_flip)
    thresh = cv2.bitwise_and(thresh, thresh_180)
    
    return thresh