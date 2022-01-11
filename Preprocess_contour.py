# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 21:48:24 2022

@author: tuank
"""
import cv2
import math


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


    