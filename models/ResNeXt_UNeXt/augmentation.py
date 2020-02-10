#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from math import log, floor, sqrt

import mxnet as mx

import numpy as np
import numpy.random as random
import cv2


# In[ ]:


class ToNDArray():
    def __call__(self, img, lbl=None):
        
#         print(img.shape)
        if len(img.shape) == 2:
            h,w = img.shape
            img = img.reshape(h,w,1)     
        img = mx.nd.array(np.moveaxis(img,-1,0))
        
        if lbl is not None:
            if len(lbl.shape) == 2:
                h,w = lbl.shape
                lbl = lbl.reshape(h,w,1)
            lbl = mx.nd.array(np.moveaxis(lbl,-1,0)) #, dtype=np.int32)
        
        return img, lbl

class Normalize:
    def __init__(self, mean, std):
        self.mean = mx.nd.array(mean)
        self.std = mx.nd.array(std)

    def __call__(self, img, lbl=None):
        img = mx.nd.transpose(img, (1, 2, 0))
        img = mx.image.color_normalize(img, self.mean, self.std)
        img = mx.nd.transpose(img, (2, 0, 1))

        return img, lbl

class AdaptNormalize:
    def __call__(self, img, lbl=None):
        avg = mx.nd.mean(img)
        std = np.std(img.asnumpy())
        img = mx.nd.transpose(img, (1, 2, 0))
        img = mx.image.color_normalize(img, avg, std)
        img = mx.nd.transpose(img, (2, 0, 1))

        return img, lbl
        
class Compose:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, img, lbl=None):
        for t in self.trans:
            img, lbl = t(img, lbl)
        return img, lbl

class AdaptResize:
    def __init__(self, resolution):
        self.resolution = resolution
        
    def __call__(self, img, lbl=None):
        
        if len(img.shape) == 3:
            nb_px = np.prod(img[:,:,0].shape)
        else:
            nb_px = np.prod(img.shape)
        factor = sqrt(nb_px / self.resolution)
        prev_h = img.shape[0]
        prev_w = img.shape[1]
        w = floor((prev_w // factor) / 32) * 32
        h = floor((prev_h // factor) / 32) * 32
#         print(w,h)
        img = cv2.resize(img, (w, h), 0, 0, cv2.INTER_LINEAR)
    
        if lbl is not None:
            lbl = cv2.resize(lbl, (w, h), 0, 0, cv2.INTER_NEAREST)
        
        return img, lbl
    
class Resize:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        
    def __call__(self, img, lbl = None):
        img = cv2.resize(img, (self.w, self.h), 0, 0, cv2.INTER_LINEAR)
        if lbl is not None:
            lbl = cv2.resize(lbl, (self.w, self.h), 0, 0, cv2.INTER_NEAREST)
        
        return img, lbl

class RandomCrop:
    def __init__(self, crop_size=None, scale=None):
        # assert min_scale <= max_scale
        self.crop_size = crop_size
        self.scale = scale
        # self.min_scale = min_scale
        # self.max_scale = max_scale

    def __call__(self, img, lbl=None):
        if self.crop_size:
            crop = self.crop_size
        else:
            crop = min(img.shape[0], img.shape[1])
        
        if crop > min(img.shape[0], img.shape[1]):
            crop = min(img.shape[0], img.shape[1])
        print(crop, img.shape[0], img.shape[1])  
        if self.scale:
            factor = random.uniform(self.scale, 1.0)
            crop = int(round(crop * factor))

        x = random.randint(0, img.shape[1] - crop)
        y = random.randint(0, img.shape[0] - crop)

        img = img[y:y+crop, x:x+crop,:]
        if lbl is not None:
            lbl = lbl[y:y+crop, x:x+crop,:]
        return img, lbl

class RandomAffine:
    def __init__(self):
        pass
    
    def __call__(self, img, lbl=None):
        #scale = random.uniform(1, 1)
        theta = random.uniform(-np.pi, np.pi)
        flipx = random.choice([-1,1])
        flipy = random.choice([-1,1])
        imgh = img.shape[0]
        imgw = img.shape[1]
        T0 = np.array([[1,0,-imgw/2.],[0,1,-imgh/2.],[0,0,1]])
        S = np.array([[flipx,0,0],[0, flipy,0],[0,0,1]])
        R = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0],[0,0,1]])
        T1 = np.array([[1,0,imgw/2.],[0,1,imgh/2.],[0,0,1]])
        M = np.dot(S, T0)
        M = np.dot(R, M)
        M = np.dot(T1, M)
        M = M[0:2,:]
        
        img = cv2.warpAffine(img, M, dsize=(imgw, imgh), flags=cv2.INTER_LINEAR)
        if lbl is not None:
            lbl = cv2.warpAffine(lbl, M, dsize=(imgw, imgh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        return img, lbl


# In[ ]:




