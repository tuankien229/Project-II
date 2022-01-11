# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 22:33:13 2022

@author: tuank
"""
import cv2
import numpy as np


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
                    
    
    return img_seg, lis_seg