#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mxnet.gluon.loss import Loss, _apply_weighting, _reshape_like


# In[ ]:


class WeightedBCEDICE(Loss):

    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, **kwargs):
        super(WeightedBCEDICE, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits

    def dice_loss(self, F, pred, label):
        smooth = 1.
        pred_y = F.argmax(pred, axis = self._axis)
        intersection = pred_y * label
        score = (2 * F.mean(intersection, axis=self._batch_axis, exclude=True) + smooth)             / (F.mean(label, axis=self._batch_axis, exclude=True) + F.mean(pred_y, axis=self._batch_axis, exclude=True) + smooth)
        
        return - F.log(score)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        if not self._from_logits:
            pred = F.log_softmax(pred, self._axis) #, self._axis
        if self._sparse_label:
            loss = -F.pick(pred, label, axis=self._axis, keepdims=True)
        else:
            label = _reshape_like(F, label, pred)
            loss = -F.sum(pred*label, axis=self._axis, keepdims=True)
#         print('before weight', F.mean(loss, axis=self._batch_axis, exclude=True))
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
#         print('after weight', F.mean(loss, axis=self._batch_axis, exclude=True))
        diceloss = self.dice_loss(F, pred, label)
        return F.mean(loss, axis=self._batch_axis, exclude=True) + diceloss


# In[ ]:


class SigmoidBinaryCrossEntropyLoss(Loss):

    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, **kwargs):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__(
            weight, batch_axis, **kwargs)
        self._from_sigmoid = from_sigmoid

    def hybrid_forward(self, F, pred, label, sample_weight=None, pos_weight=None):
        label = _reshape_like(F, label, pred)
        if not self._from_sigmoid:
            if pos_weight is None:
                # We use the stable formula: max(x, 0) - x * z + log(1 + exp(-abs(x)))
                loss = F.relu(pred) - pred * label +                     F.Activation(-F.abs(pred), act_type='softrelu')
            else:
                # We use the stable formula: x - x * z + (1 + z * pos_weight - z) * \
                #    (log(1 + exp(-abs(x))) + max(-x, 0))
                log_weight = 1 + F.broadcast_mul(pos_weight - 1, label)
                loss = pred - pred * label + log_weight *                        (F.Activation(-F.abs(pred), act_type='softrelu') + F.relu(-pred))
        else:
            eps = 1e-12
            if pos_weight is None:
                loss = -(F.log(pred + eps) * label
                         + F.log(1. - pred + eps) * (1. - label))
            else:
                loss = -(F.broadcast_mul(F.log(pred + eps) * label, pos_weight)
                         + F.log(1. - pred + eps) * (1. - label))
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


# In[ ]:


"""test"""
"""
import mxnet as mx
# p = mx.nd.array([[-10,2,1,,0,1],[0,0,1,1,0,1]])
# l = mx.nd.array([[0,0,1,1,0,1],[0,0,1,1,0,1]])
# p = mx.nd.array([[0.3],[0.9],[1]])
# l = mx.nd.array([[0],[0],[1]])
p = mx.nd.array([[0,1]])
l = mx.nd.array([[0]])
print(p)
print(l)
L = SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
los = L(p, l)
print(los)
"""


# In[ ]:




