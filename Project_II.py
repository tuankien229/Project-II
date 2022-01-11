# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 16:53:30 2021

@author: tuank
"""
#%% Funtion
# Library use in this project
import time
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
#########################################################################################################################################################
'''
This funtion used to convert nii file to image
'''
def imageData(n):
    if 0 < n < 10:
        path_seg = 'D:/Spyder/Image/MICCAI_BraTS2020_TrainingData/BraTS20_Training_00'+ str(n) + '/BraTS20_Training_00' + str(n)+ '_seg.nii.gz'
        path_t1 = 'D:/Spyder/Image/MICCAI_BraTS2020_TrainingData/BraTS20_Training_00'+ str(n) + '/BraTS20_Training_00' + str(n) + '_t1.nii.gz'
        path_t2 = 'D:/Spyder/Image/MICCAI_BraTS2020_TrainingData/BraTS20_Training_00'+ str(n) + '/BraTS20_Training_00'+ str(n) + '_t2.nii.gz'
        path_flair = 'D:/Spyder/Image/MICCAI_BraTS2020_TrainingData/BraTS20_Training_00'+ str(n) + '/BraTS20_Training_00'+ str(n) + '_flair.nii.gz'
    elif 10 <= n < 100:
        path_seg = 'D:/Spyder/Image/MICCAI_BraTS2020_TrainingData/BraTS20_Training_0'+ str(n) + '/BraTS20_Training_0' + str(n)+ '_seg.nii.gz'
        path_t1 = 'D:/Spyder/Image/MICCAI_BraTS2020_TrainingData/BraTS20_Training_0'+ str(n) + '/BraTS20_Training_0'+ str(n) + '_t1.nii.gz'
        path_t2 = 'D:/Spyder/Image/MICCAI_BraTS2020_TrainingData/BraTS20_Training_0'+ str(n) + '/BraTS20_Training_0'+ str(n) + '_t2.nii.gz'
        path_flair = 'D:/Spyder/Image/MICCAI_BraTS2020_TrainingData/BraTS20_Training_0'+ str(n) + '/BraTS20_Training_0'+ str(n) + '_flair.nii.gz'
    elif 100 <= n < 1000:
        path_seg = 'D:/Spyder/Image/MICCAI_BraTS2020_TrainingData/BraTS20_Training_'+ str(n) + '/BraTS20_Training_' + str(n)+ '_seg.nii.gz'
        path_t1 = 'D:/Spyder/Image/MICCAI_BraTS2020_TrainingData/BraTS20_Training_'+ str(n) + '/BraTS20_Training_'+ str(n) + '_t1.nii.gz'
        path_t2 = 'D:/Spyder/Image/MICCAI_BraTS2020_TrainingData/BraTS20_Training_'+ str(n) + '/BraTS20_Training_'+ str(n) + '_t2.nii.gz'
        path_flair = 'D:/Spyder/Image/MICCAI_BraTS2020_TrainingData/BraTS20_Training_'+ str(n) + '/BraTS20_Training_'+ str(n) + '_flair.nii.gz'
    
    data_seg = nib.load(path_seg).get_data()
    data_t1 = nib.load(path_t1).get_data()
    data_t2 = nib.load(path_t2).get_data()
    data_flair =  nib.load(path_flair).get_data()
    
    
    return data_seg, data_t1, data_t2, data_flair  # Return Data of each type of image


#########################################################################################################################################################
'''
This funtion used to check type of data image T2 and Flair
'''
def checkData(data, K):
    count = 0
    for i in range(data.shape[2]):
        img_og = np.array(data[:,:,i], np.uint8)
        img,_ = K_means(img_og, K)
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
        print(True)
        return True
    print(False)
    return False



#########################################################################################################################################################
'''K - means funtion'''
def K_means(img, K):
    img_copy = img.copy()
    img_copy = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    pixels = img_copy.reshape((-1,3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.01)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    result = centers[labels.flatten()]
    img_seg = result.reshape(img_copy.shape)
    img_seg = cv2.cvtColor(img_seg, cv2.COLOR_RGB2GRAY)
    lis_cen = []
    for cen in centers:
        lis_cen.append(cen[0])
    lis_cen = np.sort(lis_cen)
    lis_seg = []
    
    for c in range(len(lis_cen)):
        img_test = np.zeros(img_seg.shape, np.uint8)
        for x in range(img_seg.shape[0]):
            for y in range(img_seg.shape[1]):
                if img_seg[x,y] == lis_cen[c]:
                    img_test[x, y] = lis_cen[c]
        lis_seg.append(img_test)
                    
    
    return img_seg, lis_seg  # Return image segmentaion anh list histogram of image segmentaion



#########################################################################################################################################################
'''
This funtion used to remove small object
'''

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
def findNewContours(thresh):
    output  = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
    numLabels, labels, stats, centroids = output
    lis_cnt = []
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        lis_cnt.append([x, y, w, h])
    new_cnt = []
    while len(lis_cnt) != 0:
        if len(lis_cnt) > 1:
            lis_cnt = stackContours(lis_cnt)
        new_cnt.append(lis_cnt[0])
        lis_cnt.pop(0)
    return new_cnt
    
#########################################################################################################################################################        
def removeThresh(thresh, mask):
    _, lis_cnt = drawContours(mask)
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
def stackContours(lis_cnt):
    Xi, Yi, Wi, Hi = lis_cnt[0]
    count = 0
    new_lis = []
    lis_small = []
    for j in range(1, len(lis_cnt)):
        Xj, Yj, Wj, Hj = lis_cnt[j]
        CENTERj = (Yj + Hj//2, Xj + Wj//2)
        if Yi - 5 < CENTERj[0] <= Yi and count == 0:
            if Yi <= Yj:
                Yn = Yi
                if Yn + Hi < Yn + (Yj - Yi) + Hj:
                    Hn = Yj - Yi + Hj
                else:
                    Hn = Hi
            else:
                Yn = Yj
                if Yn + Hj < Yn + (Yi - Yj) + Hi:
                    Hn = Yi - Yj + Hi
                else:
                    Hn = Hj
            if Xi - 5 < CENTERj[1] <= Xi and count == 0:
                if Xi <= Xj:
                    Xn = Xi
                    if Xn + Wi < Xn + (Xj - Xi) + Wj:
                        Wn = Xj - Xi + Wj
                    else:
                        Wn = Wi
                else:
                    Xn = Xj
                    if Xn + Wj < Xn + (Xi - Xj) + Wi:
                        Wn = Xi - Xj + Wi
                    else:
                        Wn = Wj
                count += 1
            if Xi < CENTERj[1] <= Xi + Wi and count == 0:
                if Xi <= Xj:
                    Xn = Xi
                    if Xn + Wi < Xn + (Xj - Xi) + Wj:
                        Wn = Xj - Xi + Wj
                    else:
                        Wn = Wi
                else:
                    Xn = Xj
                    if Xn + Wj < Xn + (Xi - Xj) + Wi:
                        Wn = Xi - Xj + Wi
                    else:
                        Wn = Wj
                count += 1
            if Xi + Wi < CENTERj[1] <= Xi + Wi + 5 and count == 0:
                if Xi <= Xj:
                    Xn = Xi
                    if Xn + Wi < Xn + (Xj - Xi) + Wj:
                        Wn = Xj - Xi + Wj
                    else:
                        Wn = Wi
                else:
                    Xn = Xj
                    if Xn + Wj < Xn + (Xi - Xj) + Wi:
                        Wn = Xi - Xj + Wi
                    else:
                        Wn = Wj
                count += 1
        if Yi < CENTERj[0] <= Yi + Hi and count == 0:
            if Yi <= Yj:
                Yn = Yi
                if Yn + Hi < Yn + (Yj - Yi) + Hj:
                    Hn = Yj - Yi + Hj
                else:
                    Hn = Hi
            else:
                Yn = Yj
                if Yn + Hj < Yn + (Yi - Yj) + Hi:
                    Hn = Yi - Yj + Hi
                else:
                    Hn = Hj
            if Xi - 5 < CENTERj[1] <= Xi and count == 0:
                if Xi <= Xj:
                    Xn = Xi
                    if Xn + Wi < Xn + (Xj - Xi) + Wj:
                        Wn = Xj - Xi + Wj
                    else:
                        Wn = Wi
                else:
                    Xn = Xj
                    if Xn + Wj < Xn + (Xi - Xj) + Wi:
                        Wn = Xi - Xj + Wi
                    else:
                        Wn = Wj
                count += 1
            if Xi < CENTERj[1] <= Xi + Wi and count == 0:
                if Xi <= Xj:
                    Xn = Xi
                    if Xn + Wi < Xn + (Xj - Xi) + Wj:
                        Wn = Xj - Xi + Wj
                    else:
                        Wn = Wi
                elif Xi > Xj:
                    Xn = Xj
                    if Xn + Wj < Xn + (Xi - Xj) + Wi:
                        Wn = Xi - Xj + Wi
                    else:
                        Wn = Wj
                if Wn != Wi and Hn != Hi:
                    count += 1
                else:
                    lis_small.append(j)
            if Xi + Wi < CENTERj[1] <= Xi + Wi + 5 and count == 0:
                if Xi <= Xj:
                    Xn = Xi
                    if Xn + Wi < Xn + (Xj - Xi) + Wj:
                        Wn = Xj - Xi + Wj
                    else:
                        Wn = Wi
                else:
                    Xn = Xj
                    if Xn + Wj < Xn + (Xi - Xj) + Wi:
                        Wn = Xi - Xj + Wi
                    else:
                        Wn = Wj
                count += 1
        if Yi + Hi < CENTERj[0] <= Yi + Hi + 5 and count == 0:
            if Yi <= Yj:
                Yn = Yi
                if Yn + Hi < Yn + (Yj - Yi) + Hj:
                    Hn = Yj - Yi + Hj
                else:
                    Hn = Hi
            else:
                Yn = Yj
                if Yn + Hj < Yn + (Yi - Yj) + Hi:
                    Hn = Yi - Yj + Hi
                else:
                    Hn = Hj
            if Xi - 5 < CENTERj[1] <= Xi and count == 0:
                if Xi <= Xj:
                    Xn = Xi
                    if Xn + Wi < Xn + (Xj - Xi) + Wj:
                        Wn = Xj - Xi + Wj
                    else:
                        Wn = Wi
                else:
                    Xn = Xj
                    if Xn + Wj < Xn + (Xi - Xj) + Wi:
                        Wn = Xi - Xj + Wi
                    else:
                        Wn = Wj
                count += 1
            if Xi < CENTERj[1] <= Xi + Wi and count == 0:
                if Xi <= Xj:
                    Xn = Xi
                    if Xn + Wi < Xn + (Xj - Xi) + Wj:
                        Wn = Xj - Xi + Wj
                    else:
                        Wn = Wi
                elif Xi > Xj:
                    Xn = Xj
                    if Xn + Wj < Xn + (Xi - Xj) + Wi:
                        Wn = Xi - Xj + Wi
                    else:
                        Wn = Wj
                count += 1
            if Xi + Wi <= CENTERj[1] < Xi + Wi + 5 and count == 0:
                if Xi <= Xj:
                    Xn = Xi
                    if Xn + Wi < Xn + (Xj - Xi) + Wj:
                        Wn = Xj - Xi + Wj
                    else:
                        Wn = Wi
                else:
                    Xn = Xj
                    if Xn + Wj < Xn + (Xi - Xj) + Wi:
                        Wn = Xi - Xj + Wi
                    else:
                        Wn = Wj
                count +=1
        if count != 0:
            new_lis.append([Xn, Yn, Wn, Hn])
            lis_cnt.pop(j)
            if len(lis_small) > 0:
                lis_cnt.pop(lis_small[0])
            if len(lis_cnt) > 1:
                for i in range(1, len(lis_cnt)):
                    new_lis.append(lis_cnt[i])
                return stackContours(new_lis)
            if len(lis_cnt) == 1:
                return new_lis
    if count == 0:
        if len(lis_small) > 0:
            lis_cnt.pop(lis_small[0])
        return lis_cnt
                
                    
    
#########################################################################################################################################################
'''
This funtion used to find final thresh of image
'''
def findThresh(img, i, K):
    _, back_gr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    
    og_img = img
    _, lis_seg = K_means(og_img, K)
    seg = lis_seg[i]
    _, thresh_seg = cv2.threshold(seg, 0, 255, cv2.THRESH_BINARY)
    
    # Rotated image to find thresh then rotated again
    rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
    _, lis_seg_180 = K_means(rotated_180, K)
    seg_180 = lis_seg_180[i]
    _, thresh_180 = cv2.threshold(seg_180, 0, 255, cv2.THRESH_BINARY)
    thresh_180 = cv2.rotate(thresh_180, cv2.ROTATE_180)
    
    # Flip image to find thresh then flip agian
    flip_img = cv2.flip(img, 1)
    _, lis_seg_flip = K_means(flip_img, K)
    seg_flip = lis_seg_flip[i]
    _, thresh_flip = cv2.threshold(seg_flip, 0, 255, cv2.THRESH_BINARY)
    thresh_flip = cv2.flip(thresh_flip, 1)
    
    
    # Finaly stack all thresh image we found to find final thresh
    thresh = cv2.bitwise_and(thresh_seg, thresh_flip)
    thresh = cv2.bitwise_and(thresh, thresh_180)
    
    return thresh


#########################################################################################################################################################
def T1_process(img, check_t1):
    _, mask_inv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    if check_t1:
        thresh_3 = findThresh(img, 3, 4)
        thresh_3 = cv2.bitwise_not(thresh_3, mask = mask_inv)
        
        thresh_2 = findThresh(img, 2, 4)
        thresh_2 = cv2.bitwise_not(thresh_2, mask = mask_inv)
        
        thresh_1 = findThresh(img, 1, 4)
        
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
        
        thresh = removeThresh(thresh, mask_inv)
        
    else:
        thresh_3 = findThresh(img, 3, 4)
        
        thresh_2 = findThresh(img, 2, 4)
        thresh_2 = cv2.bitwise_not(thresh_2, mask = mask_inv)
        
        thresh_1 = findThresh(img, 1, 4)
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
        
        thresh = removeThresh(thresh, mask_inv)
    return thresh
            

#########################################################################################################################################################
def T2_process(img, check_t2, check_flair):
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    if (check_flair):
        
        thresh_3 = findThresh(img, 3, 4)
        
        thresh_2 = findThresh(img, 2, 4)
        thresh_2 = cv2.bitwise_not(thresh_2)
        
        thresh_1 = findThresh(img, 1, 4)
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
            thresh_3 = findThresh(img, 3, 4)
            thresh_3 = cv2.morphologyEx(thresh_3, cv2.MORPH_ERODE, kernel, iterations = 1)
            thresh_3 = findTumor(thresh_3, 50)
            thresh_3 = cv2.morphologyEx(thresh_3, cv2.MORPH_DILATE, kernel, iterations = 1)
            # thresh_3 = cv2.bitwise_not(thresh_3, mask = mask)
            
            thresh_2 = findThresh(img, 2, 4)
            thresh_2 = cv2.morphologyEx(thresh_2, cv2.MORPH_ERODE, kernel, iterations = 1)
            thresh_2 = findTumor(thresh_2, 50)
            thresh_2 = cv2.morphologyEx(thresh_2, cv2.MORPH_DILATE, kernel, iterations = 1)
            
            
            thresh_1 = findThresh(img, 1, 4)
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
            
            _, lis_cnt_thresh = drawContours(thresh)
            _, lis_cnt = drawContours(mask)
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
            thresh_3 = findThresh(img, 3, 4)
            
            thresh_2 = findThresh(img, 2, 4)
            thresh_2 = cv2.bitwise_not(thresh_2, mask = mask)
            
            thresh_1 = findThresh(img, 1, 4)
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
        thresh_1 = findThresh(img, 1, 4)
        
        thresh_2 = findThresh(img, 2, 4)
        thresh_2 = cv2.bitwise_not(thresh_2, mask = mask)
        
        thresh_3 = findThresh(img, 3, 4)
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
            thresh_3 = findThresh(img, 3, 4)
            thresh_3 = cv2.morphologyEx(thresh_3, cv2.MORPH_ERODE, kernel, iterations = 1)
            thresh_3 = findTumor(thresh_3, 50)
            thresh_3 = cv2.morphologyEx(thresh_3, cv2.MORPH_DILATE, kernel, iterations = 1)
            # thresh_3 = cv2.bitwise_not(thresh_3, mask = mask)
            
            thresh_2 = findThresh(img, 2, 4)
            thresh_2 = cv2.morphologyEx(thresh_2, cv2.MORPH_ERODE, kernel, iterations = 1)
            thresh_2 = findTumor(thresh_2, 50)
            thresh_2 = cv2.morphologyEx(thresh_2, cv2.MORPH_DILATE, kernel, iterations = 1)
            
            
            thresh_1 = findThresh(img, 1, 4)
            thresh_1 = cv2.bitwise_not(thresh_1, mask = mask)
            thresh_1 = cv2.morphologyEx(thresh_1, cv2.MORPH_ERODE, kernel, iterations = 1)
            thresh_1 = findTumor(thresh_1, 50)
            thresh_1 = cv2.morphologyEx(thresh_1, cv2.MORPH_DILATE, kernel, iterations = 1)
            
            thresh = cv2.bitwise_or(thresh_2, thresh_3)
            thresh = cv2.bitwise_and(thresh, thresh_1)
            
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations = 1)
            thresh = findTumor(thresh, 50)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations = 1)
            
            _, lis_cnt_thresh = drawContours(thresh)
            _, lis_cnt = drawContours(mask)
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
            
            _, lis_cnt_thresh = drawContours(thresh)
            sum_cnt_thresh = 0
            for cnt in lis_cnt_thresh:
                x, y, w, h = cnt
                sum_cnt_thresh += w*h
            
            if sum_cnt != 0:
                if sum_cnt_thresh/sum_cnt > 0.6:
                    thresh = cv2.bitwise_and(thresh_2, thresh_1)
                    _, lis_cnt_thresh = drawContours(thresh)
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
            thresh_1 = findThresh(img, 1, 4)
            
            thresh_2 = findThresh(img, 2, 4)
            thresh_2 = cv2.bitwise_not(thresh_2, mask = mask)
            
            thresh_3 = findThresh(img, 3, 4)
            
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
            _, lis_cnt_thresh = drawContours(thresh)
            _, lis_cnt = drawContours(mask)
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
def Layer_1(img_t1, img_t2, img_flair):
    global memory_cnt
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    '''----------------------------------------T2 Process----------------------------------------'''
    t2_pro = T2_process(img_t2, check_t2, check_flair)
    '''----------------------------------------Flair Process----------------------------------------'''
    flair_pro = Flair_process(img_flair, check_t2, check_flair)
    
    layer_1 = cv2.bitwise_or(t2_pro, flair_pro)
    layer_1 = cv2.morphologyEx(layer_1, cv2.MORPH_ERODE, kernel, iterations = 1)
    layer_1 = findTumor(layer_1, 50)
    layer_1 = cv2.morphologyEx(layer_1, cv2.MORPH_DILATE, kernel, iterations = 1)
    
    _, lis_cnt_layer_1 = drawContours(layer_1)
    t1_pro = T1_process(img_t1, check_t1)
    _, lis_cnt_t1 = drawContours(t1_pro)
        
    lis_cnt = findNewContours(back_t2)
    sum_cnt = 0
    for cnt in lis_cnt:
        x, y, w, h = cnt
        sum_cnt += w*h
    
    lis_layer_1, lis_t1 = checkContours(lis_cnt_layer_1, lis_cnt_t1)
    sum_cnt_t1 = 0
    for cnt_t1 in lis_t1:
        x, y, w, h = cnt_t1
        sum_cnt_t1 += w*h

    if len(lis_t1) != 0:
        memory_cnt.append(lis_t1)
        lis_t1, _ = checkContours(lis_cnt_t1, memory_cnt[-1])
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
    
    lis_cnt_layer = findNewContours(layer_1)
    sum_cnt_layer = 0
    for cnt in lis_cnt_layer:
        x, y, w, h = cnt
        sum_cnt_layer += w*h
    if sum_cnt != 0 and sum_cnt_layer/ sum_cnt > 0.8:
        layer_1 = np.zeros(layer_1.shape, np.uint8)
    
    return t2_pro, flair_pro, t1_pro, layer_1


#########################################################################################################################################################
def drawContours(thresh):
    output  = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
    numLabels, labels, stats, centroids = output
    lis_cnt = []
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        lis_cnt.append([x, y, w, h])
    return thresh, lis_cnt
    

#########################################################################################################################################################
def checkContours(lis_cnt_1, lis_cnt_2):
    lis_cnt_1_new = []
    lis_cnt_2_new = []
    for i in range(len(lis_cnt_1)):
        Xi, Yi, Wi, Hi = lis_cnt_1[i]
        Centeri = (Yi + Hi//2, Xi + Wi//2)
        for j in range(len(lis_cnt_2)):
            Xj, Yj, Wj, Hj = lis_cnt_2[j]
            Centerj = (Yj + Hj//2, Xj + Wj//2)
            dist = math.sqrt((Centeri[0] - Centerj[0])**2 + (Centeri[1] - Centerj[1])**2)
            if dist < 20:
                if lis_cnt_2[j] not in lis_cnt_2_new:
                    lis_cnt_2_new.append(lis_cnt_2[j])
                if lis_cnt_1[i] not in lis_cnt_1_new:
                    lis_cnt_1_new.append(lis_cnt_1[i])
    return lis_cnt_1_new, lis_cnt_2_new
#########################################################################################################################################################
#%% Load Data
start_time = time.time()
data_seg, data_t1, data_t2, data_flair = imageData(33)
check_t1 = checkData(data_t1, 6)
check_t2 = checkData(data_t2, 4)
check_flair = checkData(data_flair, 5)
#%% Data process
memory_cnt = []
memory_img = []
false_positive = 0
false_negative = 0
true_positive = 0
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

    
    t2_pro, flair_pro, t1_pro, layer_1 = Layer_1(img_t1, img_t2, img_flair)
    
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


