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


# In[ ]:


def _showimg(d, p, l, name, isSav = False):
    showd = d.asnumpy().astype(np.uint8) + 107
#   print(np.min(showd), np.max(showd))
    showd = np.moveaxis(showd, 0, -1)
    if isSav:
        im = Image.fromarray(showd)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(name+"_ori.jpg")
        
    plt.subplot(131)
    plt.imshow(showd, vmin=np.min(showd), vmax=np.max(showd))

    showl = l.asnumpy()
    iml = showl.astype(np.float) / np.max(showl) * 255
    if isSav:
        im = Image.fromarray(iml)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(name+"_label.jpg")
        
#   print(np.min(showl), np.max(showl))
    plt.subplot(132)
    plt.imshow(showl, vmin=0, vmax=4)

#     showp = mx.nd.softmax(p, axis=0)
    showp = mx.nd.argmax(p, axis = 0).asnumpy()
    imp = showp.astype(np.float) / np.max(showp) * 255
    if isSav:
        im = Image.fromarray(imp)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(name+"_pred.jpg")
#   showp = np.around(showp)
    print(np.min(showp), np.max(showp))
    plt.subplot(133)
    plt.imshow(showp, vmin=0, vmax=4)

    plt.show()


# In[ ]:


#_ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
_ctx = mx.cpu()

with mx.Context(_ctx):
    exp_name = "seg_iou_enp"

    train_aug = Compose([
        #RandomCrop(crop_size=5000),
        AdaptResize(720000),
        RandomAffine(),
        ToNDArray(),
        Normalize(nd.array([107]), nd.array([1]))
    ])
    test_aug = Compose([
        AdaptResize(720000),
        ToNDArray(),
        Normalize(nd.array([107]), nd.array([1]))
    ])

    # my_train = ReadDataSet('..\\dhSegment\\dataset\\ENP_500', 'train', train_aug)
    my_train = ReadDataSet('ENP_500', 'train', train_aug, fixed_weight=True)
    my_test = ReadDataSet('ENP_500', 'val', test_aug, fixed_weight=True)

    train_loader = DataLoader(my_train, batch_size=1, shuffle=False, last_batch='rollover')
    test_loader = DataLoader(my_test, batch_size=1, shuffle=False, last_batch='keep')

    model=zoo.load_pretrained_resnext_to_unext101_64_4d(ctx=_ctx,
                                                        migrate_input_norm = False,
                                                        fine_tune = False)
#     model=zoo.resume_training_unext101_64_4d(freeze_input_norm = True,
#                                              fine_tune = True,
#                                              ctx=_ctx,
#                                              symb = "unext_re_lg-symbol.json",
#                                              parame = "unext_re_lg-0000.params")
    model.hybridize()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.collect_params().initialize()

    num_epochs = 10
    num_steps = len(my_train)
    test_num_steps = len(my_test)
    # print(num_steps)

    sgd_trainer = gluon.Trainer(model.collect_params(), 'sgd', {
        'learning_rate': 0.1,
        'wd': 0.0005,
        'momentum': 0.9,
        'lr_scheduler': mx.lr_scheduler.PolyScheduler(num_steps * num_epochs, 0.1,  2, 0.0001)
    })
    adam_trainer = gluon.Trainer(model.collect_params(), 'adam',
                            {'learning_rate': 0.1,
                             'lr_scheduler': mx.lr_scheduler.PolyScheduler(num_steps * num_epochs, 
                                                                           0.1,  
                                                                           2, 
                                                                           0.0001)})

    criterion = WeightedBCEDICE(axis=1)
    # criterion = mx.gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
    
    metrics = IoUMetric(nb_cls=5, display=False, output=exp_name+'_batch_stat.txt')
    test_metrics = IoUMetric(nb_cls=5, display=False, output=exp_name+'_batch_stat.txt')

    for epoch in range(num_epochs):
        epoch_cls_mask = []
        epoch_macc_cls = []
        epoch_macc = 0
        epoch_miou_cls = []
        epoch_miou = 0
        
        t0 = time.time()
        total_loss = 0

        metrics.reset()
        
        batch_nb = 0
        counter = 0
        
        for data, label, weight, lbl_map in train_loader:

            counter += 1
            batch_size = data.shape[0]
            lbl_map = lbl_map.asnumpy()[0,]
            cls_mask = np.zeros(5)
            cls_mask[lbl_map] = 1
#             print(np.unique(weight.asnumpy()))
            with ag.record():
                pred = model(data)
#                 _showimg(data[0,:,:,:], pred[0,:,:,:], label[0,0,:,:], str(counter))
                loss = criterion(pred, label, weight)
#                 print('loss', loss)
                ag.backward(loss)
            
            total_loss += loss.asnumpy()[0]
#             print(total_loss)
            
            adam_trainer.step(batch_size)
            
            metrics.update(batch=batch_nb, labels=label, preds=pred, exist_cls=lbl_map)

            batch_nb += 1
            
            ious, miou, accs, acc = metrics.get()

            epoch_cls_mask.append(cls_mask.tolist())
            epoch_macc_cls.append(accs.tolist())
            epoch_macc += acc
            epoch_miou_cls.append(ious.tolist())
            epoch_miou += miou

        t1 = time.time()
        epoch_macc_cls = np.array(epoch_macc_cls).sum(axis=0) / np.array(epoch_cls_mask).sum(axis=0)
        epoch_macc = epoch_macc / num_steps
        epoch_miou_cls = np.array(epoch_miou_cls).sum(axis=0) / np.array(epoch_cls_mask).sum(axis=0)
        epoch_miou = epoch_miou / num_steps
        
        output = exp_name+"_train_epoch_stat.txt"
        if len(output) > 0:
            spliter = "|"
            f=open(output, "a+")
            f.write(spliter.join([str(epoch),
                                  str(t1-t0),
                                  str(total_loss/num_steps),
                                  np.array2string(epoch_macc_cls, separator=",").replace('\n', ''),
                                  str(epoch_macc),
                                  np.array2string(epoch_miou_cls, separator=",").replace('\n', ''),
                                  str(epoch_miou)]) + "\n")
            f.close()
        
#         print('epoch', epoch,
#               '\ntime', t1-t0,
#               '\navg_loss', total_loss/num_steps,
#               '\nclass-wise accuracy', epoch_macc_cls,
#               '\nmean accuracy', epoch_macc,
#               '\nclass-wise IoU', epoch_miou_cls,
#               '\nmean IoU', epoch_miou)

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
        
        for test_data, test_label, test_weight, test_lbl_map in test_loader:

            test_counter += 1

            test_lbl_map = test_lbl_map.asnumpy()[0,]
            test_cls_mask = np.zeros(5)
            test_cls_mask[test_lbl_map] = 1

            test_pred = model(test_data)
#             _showimg(test_data[0,:,:,:], test_pred[0,:,:,:], test_label[0,0,:,:], str(test_counter))
            test_loss = criterion(test_pred, test_label, test_weight)
            
            test_total_loss += test_loss.asnumpy()[0]
            
            test_metrics.update(batch=test_batch_nb, labels=label, preds=pred, exist_cls=test_lbl_map)

            test_batch_nb += 1
            
            ious, miou, accs, acc = test_metrics.get()

            test_epoch_cls_mask.append(test_cls_mask.tolist())
            test_epoch_macc_cls.append(accs.tolist())
            test_epoch_macc += acc
            test_epoch_miou_cls.append(ious.tolist())
            test_epoch_miou += miou

        test_t1 = time.time()
        test_epoch_macc_cls = np.array(epoch_macc_cls).sum(axis=0) / np.array(epoch_cls_mask).sum(axis=0)
        test_epoch_macc = epoch_macc / num_steps
        test_epoch_miou_cls = np.array(epoch_miou_cls).sum(axis=0) / np.array(epoch_cls_mask).sum(axis=0)
        test_epoch_miou = epoch_miou / num_steps
        
        test_output = exp_name+"_test_epoch_stat.txt"
        if len(test_output) > 0:
            spliter = "|"
            f=open(test_output, "a+")
            f.write(spliter.join([str(epoch),
                                  str(test_t1-test_t0),
                                  str(test_total_loss/test_num_steps),
                                  np.array2string(test_epoch_macc_cls, separator=",").replace('\n', ''),
                                  str(test_epoch_macc),
                                  np.array2string(test_epoch_miou_cls, separator=",").replace('\n', ''),
                                  str(test_epoch_miou)]) + "\n")
            f.close()
        
#         print('epoch', epoch,
#               '\ntime', test_t1-test_t0,
#               '\navg_loss', test_total_loss/test_num_steps,
#               '\nclass-wise accuracy', test_epoch_macc_cls,
#               '\nmean accuracy', test_epoch_macc,
#               '\nclass-wise IoU', test_epoch_miou_cls,
#               '\nmean IoU', test_epoch_miou)
        
    model.export('unext_re_try')


# In[ ]:




