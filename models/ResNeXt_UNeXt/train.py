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

import shutil

# In[ ]:

def _saveimgs(d, p, l, name):
    showd = (d.asnumpy()*255).astype(np.uint8)
#   print(np.min(showd), np.max(showd))
    showd = np.moveaxis(showd, 0, -1)
    
    im = Image.fromarray(showd)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save("input_image/"+name+".jpg")

    showl = l.asnumpy()
    iml = (showl*51).astype(np.uint8)
    im = Image.fromarray(iml)
    im.save("label_image/"+name+"_label.png")

    showp = mx.nd.argmax(p, axis = 0).asnumpy()
    imp = (showp*51).astype(np.uint8)
    im = Image.fromarray(imp)
    im.save("pred_image/"+name+"_pred.png")

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

# img_paths = []
# for fn in os.listdir(os.path.join('bw', 'val', 'labels')):
    # if len(fn) > 3 and fn[-4:] == '.png':
        # img_paths.append(fn[:-4])

_ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
# _ctx = mx.cpu()

# with mx.Context(_ctx):
exp_name = "seg_iou_bw_resize"

train_aug = Compose([
    #RandomCrop(crop_size=5000),
    #AdaptResize(72000),
    #AdaptResize(360000),
    RandomAffine(),
    ToNDArray(),
    #Normalize(nd.array([107]), nd.array([1]))
])
test_aug = Compose([
    #AdaptResize(72000),
    #AdaptResize(360000),
    ToNDArray(),
    #Normalize(nd.array([107]), nd.array([1]))
])

# my_train = ReadDataSet('..\\dhSegment\\dataset\\ENP_500', 'train', train_aug)
# my_train = ReadDataSet('ENP_500', 'train', train_aug, fixed_weight=True, log_weight=True, log_base=1.1)
# my_train = ReadDataSet('ENP_500', 'train', train_aug, fixed_weight=True)
# my_test = ReadDataSet('ENP_500', 'val', test_aug, fixed_weight=True)
my_train = ReadDataSet('bw', 'train', train_aug, fixed_weight=True)
my_test = ReadDataSet('bw', 'val', test_aug, fixed_weight=True)

train_loader = DataLoader(my_train, batch_size=1, shuffle=False, last_batch='rollover',num_workers=4,thread_pool=True)
test_loader = DataLoader(my_test, batch_size=1, shuffle=False, last_batch='keep',num_workers=4,thread_pool=True)

model=zoo.load_pretrained_resnext_to_unext101_64_4d(ctx=_ctx,
                                                    migrate_input_norm = False,
                                                    fine_tune = False)
# model=zoo.resume_training_unext101_64_4d(freeze_input_norm = True,
                                         # fine_tune = True,
                                         # ctx=_ctx,
                                         # symb = "unext_resize_ver03_101_64_4_px_global_weight_highest_inv_weight_enp-symbol.json",
                                         # parame = "unext_resize_ver03_101_64_4_px_global_weight_highest_inv_weight_enp-0000.params")
# model=zoo.resume_training_unext101_64_4d_beyond_word(freeze_input_norm = True,
                                                     # fine_tune = True,
                                                     # ctx=_ctx,
                                                     # symb = "unext101_64_4d_deconv_enp_72000-symbol.json",
                                                     # parame = "unext101_64_4d_deconv_enp_72000-0000.params")

with mx.Context(_ctx):
    model.hybridize()
    # model.collect_params().initialize()
    # sx = mx.sym.var('data')
    # sym = model(sx)
    # graph = mx.viz.plot_network(sym)
    # graph.format = 'tif'
    # graph.render('model')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.collect_params().initialize()

num_epochs = 80
num_steps = len(my_train)
test_num_steps = len(my_test)
# print(num_steps)

sgd_trainer = gluon.Trainer(model.collect_params(), 'sgd', {
    'learning_rate': 0.1,
    'wd': 0.0005,
    'momentum': 0.9,
    'lr_scheduler': mx.lr_scheduler.PolyScheduler(num_steps * num_epochs, 0.1,  2, 0.0001)
})
# sgd_trainer = gluon.Trainer(model.collect_params(), 'sgd', {
    # 'learning_rate': 0.1,
    # 'wd': 0.0005,
    # 'momentum': 0.9
# })
# adam_trainer = gluon.Trainer(model.collect_params(), 'adam',
                        # {'learning_rate': 0.9,
                         # 'lr_scheduler': mx.lr_scheduler.PolyScheduler(num_steps * num_epochs, 
                                                                       # 0.9,  
                                                                       # 2, 
                                                                       # 0.0001)})
adam_trainer = gluon.Trainer(model.collect_params(), 'adam',
                        {'learning_rate': 0.3})
criterion = WeightedBCEDICE(axis=1)
#criterion = mx.gluon.loss.SoftmaxCrossEntropyLoss(axis=1)

metrics = IoUMetric(nb_cls=6, display=False, output='')
test_metrics = IoUMetric(nb_cls=6, display=False, output='')
highest=0
for epoch in range(num_epochs):
    epoch_cls_mask = []
    epoch_macc_cls = []
    epoch_macc = 0
    epoch_miou_cls = []
    epoch_miou = 0
    print('epoch',epoch)
    t0 = time.time()
    total_loss = 0

    metrics.reset()
    
    batch_nb = 0
    counter = 0
    
    for data, label, weight, lbl_map in train_loader:
        data = data.as_in_context(_ctx)
        label = label.as_in_context(_ctx)
        weight = weight.as_in_context(_ctx)
        #weight = weight**(1/(2**counter))
        lbl_map = lbl_map.as_in_context(_ctx)
        counter += 1
        batch_size = data.shape[0]
        lbl_map = lbl_map.asnumpy()[0,]
        cls_mask = np.zeros(6)
        cls_mask[lbl_map] = 1
#             print(np.unique(weight.asnumpy()))
        with ag.record():
            pred = model(data)
#                 _showimg(data[0,:,:,:], pred[0,:,:,:], label[0,0,:,:], str(counter))
            loss = criterion(pred, label, weight)
#                 print('loss', loss)
            ag.backward(loss)
        
        #total_loss += loss.asnumpy()[0]
        total_loss += loss.asnumpy().mean()
#             print(total_loss)
        
        sgd_trainer.step(batch_size)
        
        metrics.update(batch=batch_nb, labels=label, preds=pred, exist_cls=lbl_map)

        batch_nb += 1
        
        ious, miou, accs, acc = metrics.get()
        # print(loss.asnumpy().mean())
        # print(mx.nd.mean(loss, axis=0, exclude=True))
        batch_output = exp_name+'_batch_stat.txt'
        if len(batch_output) > 0:
            spliter = "|"
            f=open(batch_output, "a+")
            f.write(spliter.join([str(batch_nb),
                                  str(loss.asnumpy().mean()),
                                  np.array2string(accs, separator=",").replace('\n', ''),
                                  str(acc),
                                  np.array2string(ious, separator=",").replace('\n', ''),
                                  str(miou)]) + "\n")
            f.close()

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
    
    #fid = 0

    for test_data, test_label, test_weight, test_lbl_map in test_loader:
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
        # fname = img_paths[fid]
        # fid = fid + 1
        # _saveimgs(test_data[0,:,:,:], test_pred[0,:,:,:], test_label[0,0,:,:], fname)
        test_loss = criterion(test_pred, test_label, test_weight)
        
        #test_total_loss += test_loss.asnumpy()[0]
        test_total_loss += test_loss.asnumpy().mean()
        
        test_metrics.update(batch=test_batch_nb, labels=test_label, preds=test_pred, exist_cls=test_lbl_map)

        test_batch_nb += 1
        
        ious, miou, accs, acc = test_metrics.get()
        
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
    if test_epoch_miou > highest:
        highest = test_epoch_miou
        print(highest)
        model.export('unext101_64_4d_deconv_bw_resize_72000')
        # shutil.rmtree("input_image_hi")
        # shutil.rmtree("label_image_hi")
        # shutil.rmtree("pred_image_hi")
        # os.rename("input_image", "input_image_hi")
        # os.rename("label_image", "label_image_hi")
        # os.rename("pred_image", "pred_image_hi")
        # os.mkdir("input_image")
        # os.mkdir("label_image")
        # os.mkdir("pred_image")
        
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
    
    # print('epoch', epoch,
          # '\ntime', test_t1-test_t0,
          # '\navg_loss', test_total_loss/test_num_steps,
          # '\nclass-wise accuracy', test_epoch_macc_cls,
          # '\nmean accuracy', test_epoch_macc,
          # '\nclass-wise IoU', test_epoch_miou_cls,
          # '\nmean IoU', test_epoch_miou)
    



# In[ ]:




