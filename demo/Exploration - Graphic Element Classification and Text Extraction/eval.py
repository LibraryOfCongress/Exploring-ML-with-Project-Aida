#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from math import log, floor, sqrt

import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as ag

from mxnet.gluon.data import Dataset, DataLoader
from mxnet import image

from mxnet.gluon.data.vision import transforms

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import numpy.random as random
import cv2

import time

import dl_model as zoo
from augmentation import ToNDArray, Normalize, Compose, AdaptResize, Resize, RandomCrop, RandomAffine
from losses import WeightedBCEDICE
from dataloader import ReadDataSet
from metrics import IoUMetric

import warnings

import configparser

# In[ ]:

cfg = configparser.ConfigParser()
cfg.read(config.ini)
print_predictions = cfg['project2']['print_predictions'].lower() in ['true']

def _saveimgs(d, p, l, name):
    showd = (d.asnumpy()*255).astype(np.uint8)
#   print(np.min(showd), np.max(showd))
    showd = np.moveaxis(showd, 0, -1)
    
    im = Image.fromarray(showd)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save("input_image_/"+name+".jpg")

    showl = l.asnumpy()
    iml = (showl*51).astype(np.uint8)
    im = Image.fromarray(iml)
    im.save("label_image_/"+name+"_label.png")

    showp = mx.nd.argmax(p, axis = 0).asnumpy()
    imp = (showp*51).astype(np.uint8)
    im = Image.fromarray(imp)
    im.save("pred_image_/"+name+"_pred.png")



# In[ ]:


_ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
# _ctx = mx.cpu()

# with mx.Context(_ctx):
exp_name = "seg_iou_bw_eval"

test_aug = Compose([
    #AdaptResize(72000),
    #AdaptResize(360000),
    ToNDArray(),
    #Normalize(nd.array([107]), nd.array([1]))
])

my_test = ReadDataSet('bw', 'val', test_aug, fixed_weight=True)

#train_loader = DataLoader(my_train, batch_size=1, shuffle=False, last_batch='rollover',num_workers=4,thread_pool=True)
test_loader = DataLoader(my_test, batch_size=1, shuffle=False, last_batch='keep',num_workers=4,thread_pool=True)

model=zoo.eval_model(_ctx,
                     symb = "unext101_64_4d_deconv_bw_direct_72000-symbol.json",
                     param = "unext101_64_4d_deconv_bw_direct_72000-0000.params")

with mx.Context(_ctx):
    model.hybridize()

test_num_steps = len(my_test)
# print(num_steps)
criterion = WeightedBCEDICE(axis=1)
test_metrics = IoUMetric(nb_cls=6, display=False, output='')
highest=0

# test
test_epoch_cls_mask = []
test_epoch_macc_cls = []
test_epoch_macc = 0
test_epoch_miou_cls = []
test_epoch_miou = 0

test_t0 = time.time()
test_total_loss = 0

test_metrics.reset()

test_batch_nb = 0
test_counter = 0

img_paths = []
for fn in os.listdir(os.path.join('bw', 'val', 'labels')):
    if len(fn) > 3 and fn[-4:] == '.png':
        img_paths.append(fn[:-4])

fid = 0
for test_data, test_label, test_weight, test_lbl_map in test_loader:
    single_test_time0 = time.time()
    fname = img_paths[fid]
    fid = fid + 1
    test_data = test_data.as_in_context(_ctx)
    test_label = test_label.as_in_context(_ctx)
    test_weight = test_weight.as_in_context(_ctx)
    test_lbl_map = test_lbl_map.as_in_context(_ctx)
    test_counter += 1

    test_lbl_map = test_lbl_map.asnumpy()[0,]
    test_cls_mask = np.zeros(6)
    test_cls_mask[test_lbl_map] = 1
    
    with ag.predict_mode():
        test_pred = model(test_data)
    
    if print_predictions == true:
        _saveimgs(test_data[0,:,:,:], test_pred[0,:,:,:], test_label[0,0,:,:], fname)
    test_loss = criterion(test_pred, test_label, test_weight)
    
    #test_total_loss += test_loss.asnumpy()[0]
    test_total_loss += test_loss.asnumpy().mean()
    
    test_metrics.update(batch=test_batch_nb, labels=test_label, preds=test_pred, exist_cls=test_lbl_map)

    test_batch_nb += 1
    
    ious, miou, accs, acc = test_metrics.get()
    single_test_time1 = time.time()
    test_batch_output = exp_name+'_test_batch_stat.txt'
    if len(test_batch_output) > 0:
        spliter = "|"
        f=open(test_batch_output, "a+")
        f.write(spliter.join([str(test_batch_nb),
                              str(test_loss.asnumpy()[0]),
                              np.array2string(accs, separator=",").replace('\n', ''),
                              str(acc),
                              np.array2string(ious, separator=",").replace('\n', ''),
                              str(miou)]) + "\n")
        f.close()
    print('[' + str(fid+1) + '|' + str(test_num_steps) + ']' + 'Average Accuracy: ' + str(acc) + 'Mean IoU' + str(miou) + 'Time Duration: ' + str(single_test_time1-single_test_time0))
    test_epoch_cls_mask.append(test_cls_mask.tolist())
    test_epoch_macc_cls.append(accs.tolist())
    test_epoch_macc += acc
    test_epoch_miou_cls.append(ious.tolist())
    test_epoch_miou += miou

test_t1 = time.time()
test_epoch_macc_cls = np.array(test_epoch_macc_cls).sum(axis=0) / np.array(test_epoch_cls_mask).sum(axis=0)
test_epoch_macc = test_epoch_macc / test_num_steps
test_epoch_miou_cls = np.array(test_epoch_miou_cls).sum(axis=0) / np.array(test_epoch_cls_mask).sum(axis=0)
test_epoch_miou = test_epoch_miou / test_num_steps

test_output = exp_name+"_test_epoch_stat.txt"
if len(test_output) > 0:
    spliter = "|"
    f=open(test_output, "a+")
    f.write(spliter.join([str(1),
                          str(test_t1-test_t0),
                          str(test_total_loss/test_num_steps),
                          np.array2string(test_epoch_macc_cls, separator=",").replace('\n', ''),
                          str(test_epoch_macc),
                          np.array2string(test_epoch_miou_cls, separator=",").replace('\n', ''),
                          str(test_epoch_miou)]) + "\n")

print('Average test performance',
      '\ntime', test_t1-test_t0,
      '\navg_loss', test_total_loss/test_num_steps,
      '\nclass-wise accuracy', test_epoch_macc_cls,
      '\nmean accuracy', test_epoch_macc,
      '\nclass-wise IoU', test_epoch_miou_cls,
      '\nmean IoU', test_epoch_miou)
    



# In[ ]:




