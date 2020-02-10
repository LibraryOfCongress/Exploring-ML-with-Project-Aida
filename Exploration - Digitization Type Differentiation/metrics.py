#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import mxnet as mx
import numpy as np

import matplotlib.pyplot as plt


# In[ ]:


class IoUMetric(mx.metric.EvalMetric):
    """CalculSegMetricate metrics for Seg training """
    def __init__(self, nb_cls, display=False, output="", eps=1e-8):
        
        self.nbcls = nb_cls
        self.eps = eps
        self.display = display
        self.output = output
        super(IoUMetric, self).__init__('Segmentation_IoU')
        self.reset()

    def reset(self):
        """
        override reset behavior
        """
        self.IoUs = np.zeros(self.nbcls)
        self.mIoU = 0
        self.acc = np.zeros(self.nbcls)
        self.macc = 0
        
    def update(self, batch, labels, preds, exist_cls):
        """
        Implementation of updating metrics
        """
        # get generated multi label from network
        
        l = labels.asnumpy().astype(np.int)[0,0,]
        p = preds.asnumpy()[0,]
        
        pl = np.argmax(p, axis=0)
            
        accs = []
        for i in range(self.nbcls):
            suml = np.sum((l == i)).astype(np.int) + self.eps
            correct = np.sum(np.logical_and((pl==i), (l==i)).astype(np.int))
            accs.append(correct/suml)

        cls_mask = np.zeros(5)
        cls_mask[exist_cls] = 1

        self.acc = np.array(accs) * cls_mask
        self.macc = self.acc.sum() / cls_mask.sum()

        self.IoUs = IoU(pred=pl, label=l, nb_cls=self.nbcls, eps=self.eps) * cls_mask
        self.mIoU = self.IoUs.sum() / cls_mask.sum()
        
        if len(self.output) > 0:
            spliter = "|"
            f=open(self.output, "a+")
            f.write(spliter.join([str(batch),
                                  np.array2string(self.acc, separator=",").replace('\n', ''),
                                  str(self.macc),
                                  np.array2string(self.IoUs, separator=",").replace('\n', ''),
                                  str(self.mIoU)]) + "\n")
            f.close()
        
        if self.display:
            print('batch', batch,
                  '\nclass-wise accuracies\n', self.acc,
                  '\nmean accuracy\n', self.macc,
                  '\nclass-wise IoU\n', self.IoUs,
                  '\nmean IoU\n', self.mIoU)
        
    def get(self):
        """Get the current evaluation result.
        Override the default behavior

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        return self.IoUs, self.mIoU, self.acc, self.macc


# In[ ]:


class ConfusMatMulticls(mx.metric.EvalMetric):
    """CalculSegMetricate metrics for Seg training """
    def __init__(self, nb_cls, display=False, output="", eps=1e-8):
        self.name = 'Confusion Matrix'
        self.nb_cls = nb_cls
        self.eps = eps
        self.display = display
        self.output = output
        super(ConfusMatMulticls, self).__init__('ConfusMatMulticls')
        self.reset()

    def reset(self):
        """
        override reset behavior
        """
        self.confusion_mat = np.zeros((self.nb_cls,self.nb_cls))
        self.tps = np.zeros(self.nb_cls)
        self.tns = np.zeros(self.nb_cls)
        self.fps = np.zeros(self.nb_cls)
        self.fns = np.zeros(self.nb_cls)

    def update(self, batch, labels, preds):
        """
        Implementation of updating metrics
        """
        # get generated multi label from network
        
        p = mx.nd.argmax(preds, axis=1, keepdims=True)
        p = p.asnumpy()
        
        count = p.shape[0]
        
        l = labels.asnumpy()
        
        confusion_mat = confusion_mat_multiclass(p,l,self.nb_cls)
        self.confusion_mat += confusion_mat
        
        tps = getTP(confusion_mat)
        tns = getTN(confusion_mat)
        fps = getFP(confusion_mat)
        fns = getFN(confusion_mat)
        
        self.tps += tps
        self.tns += tns
        self.fps += fps
        self.fns += fns
        
        acc = (tps + tns) / (tps + tns + fps + fns)
        recalls = tps / ((tps + fns) + self.eps)
        precisions = tps / ((tps + fps) + self.eps)
        f1s = 2 * (recalls * precisions) / ((recalls + precisions) + self.eps)
        
        if len(self.output) > 0:
            spliter = "|"
            f=open(self.output, "a+")
            f.write(spliter.join([str(batch),
                                  np.array2string(confusion_mat, separator=",").replace('\n', ''),
                                  np.array2string(tps, separator=",").replace('\n', ''),
                                  np.array2string(tns, separator=",").replace('\n', ''),
                                  np.array2string(fps, separator=",").replace('\n', ''),
                                  np.array2string(fns, separator=",").replace('\n', ''),
                                  np.array2string(acc, separator=",").replace('\n', ''),
                                  np.array2string(precisions, separator=",").replace('\n', ''),
                                  np.array2string(recalls, separator=",").replace('\n', ''),
                                  np.array2string(f1s, separator=",").replace('\n', '')]) + "\n")
            f.close()
        
        if self.display:
            print('batch', batch,
                  '\nconfusion matrix\n', confusion_mat,
                  '\nTP\n', tps,
                  '\nTN\n', tns,
                  '\nFP\n', fps,
                  '\nFN\n', fns,
                  '\nAccuracy', acc,
                  '\nPrecision\n', precisions,
                  '\nRecall\n', recalls,
                  '\nF1\n', f1s)
        
    def get(self):
        """Get the current evaluation result.
        Override the default behavior

        Returns
        -------
        confusion_mat : confusion matrix
        """    
        return self.confusion_mat, self.tps, self.tns, self.fps, self.fns


# In[ ]:


class ConfusMatMultilb(mx.metric.EvalMetric):
    """metrics for multi-labeling training: binary results for each class """
    def __init__(self, nb_cls, display=False, output="", eps=1e-8):
        self.name = 'Confusion Matrix'
        self.nb_cls = nb_cls
        self.eps = eps
        self.display = display
        self.output = output
        super(ConfusMatMultilb, self).__init__('ConfusMatMultilb')
        self.reset()

    def reset(self):
        """
        override reset behavior
        """
        self.confusion_mat = np.zeros((self.nb_cls,2,2))
        self.tps = np.zeros(self.nb_cls)
        self.tns = np.zeros(self.nb_cls)
        self.fps = np.zeros(self.nb_cls)
        self.fns = np.zeros(self.nb_cls)

    def update(self, batch, labels, preds):
        """
        Implementation of updating metrics
        """
        # get generated multi label from network
        
        p = preds
        p = p.asnumpy()
        
        count = p.shape[0]
        
        l = labels.asnumpy()
        
        confusion_mat = confusion_mat_multilabel(p,l,self.nb_cls)
        self.confusion_mat += confusion_mat
        
        tps = getTP_multilb(confusion_mat)
        tns = getTN_multilb(confusion_mat)
        fps = getFP_multilb(confusion_mat)
        fns = getFN_multilb(confusion_mat)
        
        self.tps += tps
        self.tns += tns
        self.fps += fps
        self.fns += fns
        
        acc = (tps + tns) / (tps + tns + fps + fns)
        recalls = tps / ((tps + fns) + self.eps)
        precisions = tps / ((tps + fps) + self.eps)
        f1s = 2 * (recalls * precisions) / ((recalls + precisions) + self.eps)
        
        if len(self.output) > 0:
            spliter = "|"
            f=open(self.output, "a+")
            f.write(spliter.join([str(batch),
                                  np.array2string(confusion_mat, separator=",").replace('\n', ''),
                                  np.array2string(tps, separator=",").replace('\n', ''),
                                  np.array2string(tns, separator=",").replace('\n', ''),
                                  np.array2string(fps, separator=",").replace('\n', ''),
                                  np.array2string(fns, separator=",").replace('\n', ''),
                                  np.array2string(acc, separator=",").replace('\n', ''),
                                  np.array2string(precisions, separator=",").replace('\n', ''),
                                  np.array2string(recalls, separator=",").replace('\n', ''),
                                  np.array2string(f1s, separator=",").replace('\n', '')]) + "\n")
            f.close()
        
        if self.display:
            print('batch', batch,
                  '\nconfusion matrix\n', confusion_mat,
                  '\nTP\n', tps,
                  '\nTN\n', tns,
                  '\nFP\n', fps,
                  '\nFN\n', fns,
                  '\nAccuracy', acc,
                  '\nPrecision\n', precisions,
                  '\nRecall\n', recalls,
                  '\nF1\n', f1s)
        
    def get(self):
        """Get the current evaluation result.
        Override the default behavior

        Returns
        -------
        confusion_mat : confusion matrix
        """    
        return self.confusion_mat, self.tps, self.tns, self.fps, self.fns


# In[ ]:


def IoU(pred, label, nb_cls, eps):
    # input must be numpy array
    # pred must has been argmax-ed
    ious = []
    for cls in range(nb_cls):
        # pick out class labels and predictions
        pred_cls = (pred == cls)
        label_cls = (label == cls)
        
        intersaction = np.logical_and(pred_cls, label_cls).astype(np.float)
        union = np.logical_or(pred_cls, label_cls).astype(np.float)
        
        ious.append(np.mean(np.sum(intersaction, axis=0))/(np.mean(np.sum(union, axis=0))+eps))
        
    return np.array(ious)
        


# In[ ]:


def mIoU(ious):
    return ious.sum() / (ious!=0).sum()


# In[ ]:


# tp, tn, fp, fn class-wise
def tp_tn_fp_fn_cls(pred, label, nb_cls, batch_axis=0):
    nb_axis = len(pred.shape)
    axis = ()
    for ax in range(1, nb_axis):
        axis = axis + (ax,)
        
    tps = []
    tns = []
    fps = []
    fns = []
    
    for cls in range(nb_cls):
        pred_cls = (pred == cls)
        label_cls = (label == cls)
        
        tp = np.sum(np.logical_and(pred_cls, label_cls).astype(np.int64), axis=axis)
        diff = np.subtract(pred_cls.astype(np.int), label_cls.astype(np.int))
        batch_sz = diff.shape[0]
#         print("diff",diff)
        fn = []
        fp = []
        tn = []
        for i in range(batch_sz):
            uniq, counts = np.unique(diff[i,], return_counts=True)
            dic = dict(zip(uniq, counts))
#             print(dic)
            if -1 in dic.keys():
                fn.append(dic[-1])
            else:
                fn.append(0)
                
            if 1 in dic.keys():
                fp.append(dic[1])
            else:
                fp.append(0)
            
            if 0 in dic.keys():
                tn.append(dic[0])
            else:
                tn.append(0)

        fn = np.array(fn)
        fp = np.array(fp)
        tn = np.array(tn) - tp
        
        tps.append(tp)
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)
        
    return np.array(tps),np.array(tns),np.array(fps),np.array(fns)
        


# In[ ]:


def tp_tn_fp_fn_avg(tps, tns, fps, fns):
    tp_avg = np.sum(np.sum(tps, axis=0)/tps.shape[0])/tps.shape[1]
    tn_avg = np.sum(np.sum(tns, axis=0)/tns.shape[0])/tns.shape[1]
    fp_avg = np.sum(np.sum(fps, axis=0)/fps.shape[0])/fps.shape[1]
    fn_avg = np.sum(np.sum(fns, axis=0)/fns.shape[0])/fns.shape[1]
    return tp_avg, tn_avg, fp_avg, fn_avg


# In[ ]:


def tp_tn_fp_fn_sum(tps, tns, fps, fns):
    tp_sum = np.sum(tps)
    tn_sum = np.sum(tns)
    fp_sum = np.sum(fps)
    fn_sum = np.sum(fns)
    return tp_sum, tn_sum, fp_sum, fn_sum


# In[ ]:


def confusion_mat_multiclass(pred, label, nbCls):
    record = np.zeros((nbCls,nbCls))
    
    # pred and label must have the same shape of n by 1
    
    batchsz = pred.shape[0]
    for item in range(batchsz):
#         print(item)
#         print('label', label[item], 'pred', pred[item])
        record[label[item].astype(np.int), pred[item].astype(np.int)]  += 1
        
    return record


# In[ ]:


def confusion_mat_multilabel(pred, label, nbCls):
    rec = np.zeros((nbCls, 2,2))
    
    for i in range(nbCls):
        rec[i,:,:] = confusion_mat_multiclass(pred[:,i,0], label[:,i,0], 2)
        
    return rec


# In[ ]:


def getTP(cm):
    return np.diagonal(cm)


# In[ ]:


def getTP_bin(cm):
    return cm[1,1]


# In[ ]:


def getTP_multilb(cms):
    tps = np.zeros(cms.shape[0])
    for i in range(cms.shape[0]):
        tps[i] = getTP_bin(cms[i,:,:])
        
    return tps


# In[ ]:


def getTN(cm):
    tps = getTP(cm)
    tns = []
    for i in range(cm.shape[0]):
        tns.append(np.sum(tps) - tps[i])
        
    return np.array(tns)


# In[ ]:


def getTN_bin(cm):
    return cm[0,0]


# In[ ]:


def getTN_multilb(cms):
    tns = np.zeros(cms.shape[0])
    for i in range(cms.shape[0]):
        tns[i] = getTN_bin(cms[i,:,:])
        
    return tns


# In[ ]:


def getFP(cm):
    tps = getTP(cm)
    fps = []
    for i in range(cm.shape[0]):
        fps.append(np.sum(cm[i,:]) - tps[i])
    
    return np.array(fps)


# In[ ]:


def getFP_bin(cm):
    return cm[0,1]


# In[ ]:


def getFP_multilb(cms):
    fps = np.zeros(cms.shape[0])
    for i in range(cms.shape[0]):
        fps[i] = getFP_bin(cms[i,:,:])
        
    return fps


# In[ ]:


def getFN(cm):
    tps = getTP(cm)
    fns = []
    for i in range(cm.shape[0]):
        fns.append(np.sum(cm[:,i]) - tps[i])
    
    return np.array(fns)


# In[ ]:


def getFN_bin(cm):
    return cm[1,0]


# In[ ]:


def getFN_multilb(cms):
    fns = np.zeros(cms.shape[0])
    for i in range(cms.shape[0]):
        fns[i] = getFN_bin(cms[i,:,:])
        
    return fns


# In[ ]:


def fscore(precision, recall):
    return 2 * ((precision * recall) / np.add(precision, recall))


# In[ ]:


def precision(tp,fp):
    return np.mean(tp/(np.add(tp,fp)))


# In[ ]:


def recall(tp, fn):
    return np.mean(tp/(np.add(tp,fn)))


# In[ ]:


def show(pred, label):
    showd = dlist[i][0,:,:,:].asnumpy().astype(np.uint8) + 107
    #  print(np.min(showd), np.max(showd))
    showd = np.moveaxis(showd, 0, -1)
    plt.subplot(131)
    plt.imshow(showd, vmin=np.min(showd), vmax=np.max(showd))

    showl = llist[i][0,0,:,:].asnumpy()
    #  print(np.min(showl), np.max(showl))
    plt.subplot(132)
    plt.imshow(showl, vmin=0, vmax=4)

    showp = mx.nd.softmax(preds[i][0,:,:,:], axis=0)
    showp = mx.nd.argmax(showp, axis = 0).asnumpy()
    #showp = np.around(showp)
    print(np.min(showp), np.max(showp))
    plt.subplot(133)
    plt.imshow(showp, vmin=0, vmax=4)

    plt.show()

