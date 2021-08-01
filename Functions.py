"""
This file contains two functions which are feature_extractor and segmentation.
feature extractor function is the proposed feature extractor algorithm and
the segmentation function is actually the nucleus segmentation algorithm
used in the paper.
"""

import time
import cv2
import numpy as np
import json
import csv
import random
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
import pyhdust.images as phim
from scipy.spatial import ConvexHull
import glob
import joblib
from skimage.feature import greycomatrix, greycoprops
from skimage import filters as fl


def segmentation(img):
    """
    :param img: input rgb image
    :param min_area: minimum area of nucleus, if area of nucleus is lower than this value, this means
            that the nucleus is not detected
    :return: binary of nucleus, binary of convexhull, binary of ROC
    """
    org_img = img.copy()

    # Color balancing
    Gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]

    mean_gray = np.mean(Gray)
    mean_R = np.mean(R)
    mean_G = np.mean(G)
    mean_B = np.mean(B)

    R_ = R * (mean_gray / mean_R)
    G_ = G * (mean_gray / mean_G)
    B_ = B * (mean_gray / mean_B)

    R_[R_ > 255] = 255
    G_[G_ > 255] = 255
    B_[B_ > 255] = 255

    balance_img = np.zeros_like(org_img)
    balance_img[:, :, 0] = R_.copy()
    balance_img[:, :, 1] = G_.copy()
    balance_img[:, :, 2] = B_.copy()

    # >>>>>> 8 ms <<<<<<

    # balance_img = org_img.copy()
    cmyk = phim.rgb2cmyk(balance_img)
    _M = cmyk[:, :, 1]
    _K = cmyk[:, :, 3]

    _S = cv2.cvtColor(balance_img, cv2.COLOR_RGB2HLS_FULL)[:, :, 2]

    min_MS = np.minimum(_M, _S)
    a_temp = np.where(_K < _M, _K, _M)
    KM = _K - a_temp

    b_temp = np.where(min_MS < KM, min_MS, KM)
    min_MS_KM = min_MS - b_temp

    # cv2.imshow('Step 1' , cv2.resize(Nucleus_img , (256 ,256)))

    # Step 2 :
    min_MS_KM = cv2.GaussianBlur(min_MS_KM, ksize=(5, 5), sigmaX=0)
    try:
        thresh2 = fl.threshold_multiotsu(min_MS_KM, 2)
        Nucleus_img = np.zeros_like(min_MS_KM)
        Nucleus_img[min_MS_KM >= thresh2] = 255
    except:
        print('try-Except')
        _M = cv2.GaussianBlur(_M, ksize=(5, 5), sigmaX=0)
        thresh2 = fl.threshold_multiotsu(_M, 2)
        Nucleus_img = np.zeros_like(_M)
        Nucleus_img[_M >= thresh2] = 255

    _, contours, _ = cv2.findContours(Nucleus_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pad_del = np.zeros_like(Nucleus_img)

    max_area = max(cv2.contourArea(contours[idx]) for idx in np.arange(len(contours)))
    for j in range(len(contours)):
        if cv2.contourArea(contours[j]) < (max_area / 10):
            cv2.drawContours(pad_del, contours, j, color=255, thickness=-1)
    Nucleus_img[pad_del > 0] = 0

    _, contours, _ = cv2.findContours(Nucleus_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _perimeter = 0
    for cnt in contours:
        _perimeter += cv2.arcLength(cnt, True)

    temp_points = np.argwhere(Nucleus_img == 255)
    Ncl_points = np.zeros_like(temp_points)
    Ncl_points[:, 0] = temp_points[:, 1]
    Ncl_points[:, 1] = temp_points[:, 0]
    _area = np.sum(Nucleus_img)

    cvx_hull = ConvexHull(Ncl_points)
    Cvx_area = cvx_hull.volume
    Cvx_prm = cvx_hull.area
    Verc = cvx_hull.vertices
    Corners = []
    for idx in range(len(Verc)):
        tempcol = Ncl_points[Verc[idx], 0]
        temprow = Ncl_points[Verc[idx], 1]
        Corners.append([tempcol, temprow])
    Corners = np.array(Corners)
    Corners = np.reshape(Corners, newshape=(Corners.shape[0], 1, 2))

    img_convex = np.zeros_like(Nucleus_img)
    cv2.drawContours(img_convex, [Corners], 0, color=255, thickness=-1)
    CVX_points = np.argwhere(img_convex == 255)

    img_ROC = img_convex - Nucleus_img

    return Nucleus_img, img_convex, img_ROC



def feature_extractor(img, min_area=100):

    Ftr_List = []
    #org_img = cv2.resize(img, dsize=(height, width))
    org_img = img.copy()
    img[:, :, 0] = org_img[:, :, 0].copy()
    img[:, :, 1] = org_img[:, :, 1].copy()
    img[:, :, 2] = org_img[:, :, 2].copy()

    Gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]

    mean_gray = np.mean(Gray)
    mean_R = np.mean(R)
    mean_G = np.mean(G)
    mean_B = np.mean(B)

    R_ = R * (mean_gray / mean_R)
    G_ = G * (mean_gray / mean_G)
    B_ = B * (mean_gray / mean_B)

    R_[R_ > 255] = 255
    G_[G_ > 255] = 255
    B_[B_ > 255] = 255

    balance_img = np.zeros_like(org_img)
    balance_img[:, :, 0] = R_.copy()
    balance_img[:, :, 1] = G_.copy()
    balance_img[:, :, 2] = B_.copy()

    # >>>>>> 8 ms <<<<<<

    # balance_img = org_img.copy()
    cmyk = phim.rgb2cmyk(balance_img)
    _M = cmyk[:, :, 1]
    _K = cmyk[:, :, 3]

    _S = cv2.cvtColor(balance_img, cv2.COLOR_RGB2HLS_FULL)[:, :, 2]

    min_MS = np.minimum(_M, _S)
    a_temp = np.where(_K < _M, _K, _M)
    KM = _K - a_temp

    b_temp = np.where(min_MS < KM, min_MS, KM)
    min_MS_KM = min_MS - b_temp

    # cv2.imshow('Step 1' , cv2.resize(Nucleus_img , (256 ,256)))

    # Step 2 :
    min_MS_KM = cv2.GaussianBlur(min_MS_KM, ksize=(5, 5), sigmaX=0)
    try:
        thresh2 = fl.threshold_multiotsu(min_MS_KM, 2)
        Nucleus_img = np.zeros_like(min_MS_KM)
        Nucleus_img[min_MS_KM >= thresh2] = 255
    except:
        print('try-Except')
        _M = cv2.GaussianBlur(_M, ksize=(5, 5), sigmaX=0)
        thresh2 = fl.threshold_multiotsu(_M, 2)
        Nucleus_img = np.zeros_like(_M)
        Nucleus_img[_M >= thresh2] = 255

    _, contours, _ = cv2.findContours(Nucleus_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pad_del = np.zeros_like(Nucleus_img)

    max_area = max(cv2.contourArea(contours[idx]) for idx in np.arange(len(contours)))
    for j in range(len(contours)):
        if cv2.contourArea(contours[j]) < (max_area / 10):
            cv2.drawContours(pad_del, contours, j, color=255, thickness=-1)
    Nucleus_img[pad_del > 0] = 0

    _, contours, _ = cv2.findContours(Nucleus_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _perimeter = 0
    for cnt in contours:
        _perimeter += cv2.arcLength(cnt, True)

    temp_points = np.argwhere(Nucleus_img==255)
    Ncl_points = np.zeros_like(temp_points)
    Ncl_points[:,0] = temp_points[:,1]
    Ncl_points[:,1] = temp_points[:,0]
    _area = np.sum(Nucleus_img)

    cvx_hull = ConvexHull(Ncl_points)
    Cvx_area = cvx_hull.volume
    Cvx_prm = cvx_hull.area
    Verc = cvx_hull.vertices
    Corners = []
    for idx in range(len(Verc)):
        tempcol = Ncl_points[Verc[idx], 0]
        temprow = Ncl_points[Verc[idx], 1]
        Corners.append([tempcol, temprow])
    Corners = np.array(Corners)
    Corners = np.reshape(Corners, newshape=(Corners.shape[0], 1, 2))

    img_convex = np.zeros_like(Nucleus_img)
    cv2.drawContours(img_convex, [Corners], 0, color=255, thickness=-1)
    CVX_points = np.argwhere(img_convex == 255)

    img_ROC = img_convex - Nucleus_img
    ROC_points = np.argwhere(img_ROC == 255)

    flag_empty = len(contours) > 0
    if not flag_empty:
        Error = '[Error 1]: No contours are detected'
        print(Error)
        return False, Error, None

    if max_area <= min_area:
        Error = '[ERROR 2]: max area of nucleus is lower than %d'%(min_area)
        print(Error)
        return False, Error, None

    Circularity = (_perimeter) ** 2 / (4 * 3.14 * _area)
    Convexity = (Cvx_prm / _perimeter)
    Solidity = (_area / Cvx_area)
    Shape_Features = np.array([Circularity, Convexity, Solidity])
    Ftr_List.extend([Circularity, Convexity, Solidity])
    if np.sum(img_convex - Nucleus_img) == 0:
        print('******* Convex image == nucleus_image ********')
        temp = [1]*72
        temp.extend(Ftr_List)
        return True, None, np.array(temp)
    # >>>>>> NEW CODES <<<<<<<<<
    ALL_Channels = []
    ALL_Channels.append(balance_img[:, :, 0]) # channel R : index 0
    ALL_Channels.append(balance_img[:, :, 1]) # channel G : index 1
    ALL_Channels.append(balance_img[:, :, 2]) # channel B : index 2

    HSV = cv2.cvtColor(balance_img, cv2.COLOR_RGB2HSV)
    ALL_Channels.append(HSV[:, :, 0]) # channel H : index 3
    ALL_Channels.append(HSV[:, :, 1]) # channel S : index 4
    ALL_Channels.append(HSV[:, :, 2]) # channel V : index 5

    LAB = cv2.cvtColor(balance_img, cv2.COLOR_RGB2LAB)
    ALL_Channels.append(LAB[:, :, 0]) # channel L : index 6
    ALL_Channels.append(LAB[:, :, 1]) # channel A : index 7
    ALL_Channels.append(LAB[:, :, 2]) # channel BB : index 8

    YCrCb = cv2.cvtColor(balance_img, cv2.COLOR_RGB2YCrCb)
    ALL_Channels.append(YCrCb[:, :, 0]) # channel Y : index 9
    ALL_Channels.append(YCrCb[:, :, 1]) # channel Cr : index 10
    ALL_Channels.append(YCrCb[:, :, 2]) # channel Cb : index 11

    NCL_pxls_value = np.zeros(shape=(len(ALL_Channels), Ncl_points.shape[0]), dtype=np.uint8) # intensity of nucleus
    CVX_pxls_Value = np.zeros(shape=(len(ALL_Channels), CVX_points.shape[0]), dtype=np.uint8) # intensity of convexhull
    ROC_pxls_Value = np.zeros(shape=(len(ALL_Channels), ROC_points.shape[0]), dtype=np.uint8) # intensity of ROC

    for ch in range(len(ALL_Channels)):
        p_roc, p_ncl = 0, 0
        for p in range(CVX_points.shape[0]):
            row, col = CVX_points[p, 0], CVX_points[p, 1]
            CVX_pxls_Value[ch, p] = ALL_Channels[ch][row, col]

            if Nucleus_img[row, col] == 255:
                NCL_pxls_value[ch, p_ncl] = ALL_Channels[ch][row, col]
                p_ncl += 1
            else:
                ROC_pxls_Value[ch, p_roc] = ALL_Channels[ch][row, col]
                p_roc += 1

    Ncl_mean_std = np.zeros(shape=(len(ALL_Channels), 2), dtype=np.float) # mean and std for nucleus in all channels
    Ncl_mean_std[:, 0] = np.mean(NCL_pxls_value, axis=1)
    Ncl_mean_std[:, 1] = np.std(NCL_pxls_value, axis=1)

    Cvx_mean_std = np.zeros(shape=(len(ALL_Channels), 2), dtype=np.float) # mean and std for convexhull in all channels
    Cvx_mean_std[:, 0] = np.mean(CVX_pxls_Value, axis=1)
    Cvx_mean_std[:, 1] = np.std(CVX_pxls_Value, axis=1)

    Roc_mean_std = np.zeros(shape=(len(ALL_Channels), 2), dtype=np.float) # mean and std for ROC in all channels
    Roc_mean_std[:, 0] = np.mean(ROC_pxls_Value, axis=1)
    Roc_mean_std[:, 1] = np.std(ROC_pxls_Value, axis=1)

    Ratio_Ncl2Cvx = np.reshape(np.divide(Ncl_mean_std, Cvx_mean_std), newshape=(len(ALL_Channels)*2,))
    Ratio_Roc2Cvx = np.reshape(np.divide(Roc_mean_std, Cvx_mean_std), newshape=(len(ALL_Channels)*2,))
    # Ratio_Roc2Ncl = np.reshape(np.divide(Roc_mean_std, Ncl_mean_std), newshape=(len(ALL_Channels)*2,))
    Color_Features = np.concatenate((Ratio_Ncl2Cvx, Ratio_Roc2Cvx))
    #Color_Features = np.nan_to_num(Color_Features, nan=0, posinf=1)
    ALL_Features = np.concatenate((Color_Features, Shape_Features))

    return True, None, ALL_Features

