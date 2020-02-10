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
import numpy as np
import numpy.random as random
import cv2

import time

import dl_model as zoo
from augmentation import ToNDArray, Normalize, Compose, AdaptResize, Resize, RandomCrop, RandomAffine
from losses import WeightedBCEDICE
from dataloader import ReadDataSet_micro
from metrics import ConfusMatMulticls

import warnings


# In[ ]:


_ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
# _ctx = mx.cpu()

test_aug = Compose([
    Resize(224,224),
    ToNDArray(),
    Normalize(nd.array([107]), nd.array([1]))
])

my_eval = ReadDataSet_micro('dataset', 'test', 2, _ctx, test_aug)

test_loader = DataLoader(my_eval, batch_size=10, shuffle=True, last_batch='keep')

model=zoo.eval_model(_ctx,
                     symb = "micro_highscore_randAffine_2nd-symbol.json",
                     param = "micro_highscore_randAffine_2nd-0000.params")

with mx.Context(_ctx):
    model.hybridize()

num_epochs = 1
num_steps = len(my_train)

# criterion = WeightedBCEDICE(axis=1)
criterion = mx.gluon.loss.SoftmaxCrossEntropyLoss()

# test
test_t0 = time.time()
test_total_loss = 0
test_metri.reset()

test_count = 0
test_nbatch = 0
for data, label in test_loader:
    batch_size = data.shape[0]
    preds = model(data)
    losses = criterion(preds, label)
    
    test_total_loss += sum([l.sum().asscalar() for l in losses])

    test_metri.update(batch=test_nbatch, labels=label, preds=preds)

    test_count += batch_size
    test_nbatch += 1

test_confusionMat, test_tps, test_tns, test_fps, test_fns = test_metri.get()

test_acc = (test_tps + test_tns) / (test_tps + test_tns + test_fps + test_fns)
test_recalls = test_tps / ((test_tps + test_fns) + 1e-8)
test_precisions = test_tps / ((test_tps + test_fps) + 1e-8)
test_f1s = 2 * (test_recalls * test_precisions) / ((test_recalls + test_precisions) + 1e-8)

test_t1 = time.time()

test_output = "epoch_stat_test.txt"
if len(test_output) > 0:
    spliter = "|"
    test_f=open(test_output, "a+")
    test_f.write(spliter.join([str(epoch),
                               str(test_t1-test_t0),
                               str(total_loss/test_count),
                               np.array2string(test_confusionMat, separator=",").replace('\n', ''),
                               np.array2string(test_tps, separator=",").replace('\n', ''),
                               np.array2string(test_tns, separator=",").replace('\n', ''),
                               np.array2string(test_fps, separator=",").replace('\n', ''),
                               np.array2string(test_fns, separator=",").replace('\n', ''),
                               np.array2string(test_acc, separator=",").replace('\n', ''),
                               np.array2string(test_precisions, separator=",").replace('\n', ''),
                               np.array2string(test_recalls, separator=",").replace('\n', ''),
                               np.array2string(test_f1s, separator=",").replace('\n', '')]) + "\n")
    test_f.close()

print('average performance',
      '\ntime', test_t1-test_t0,
      '\navg_loss', test_total_loss/test_count,
      '\nconfusion matrix\n', test_confusionMat,
      '\nTP', test_tps,
      '\nTN', test_tns,
      '\nFP', test_fps,
      '\nFN', test_fns,
      '\nAccuracy', test_acc,
      '\nPrecision', test_precisions,
      '\nRecall', test_recalls,
      '\nF1', test_f1s)
    
model.export('micro_1st')


# In[ ]:




