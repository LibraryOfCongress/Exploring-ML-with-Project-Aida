#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mxnet as mx
from mxnet.gluon.nn import HybridBlock
from mxnet import ndarray as F


# In[2]:


class GlobalMinPool2D(HybridBlock):
    def __init__(self, layout='NCHW', **kwargs):
        super(GlobalMinPool2D, self).__init__(**kwargs)
        self.layout = layout
        
    def hybrid_forward(self, F, x):
        x = F.min(data=x, axis=(2,3), keepdims=True, name='GlobalMinPool2D')
        return x


# In[ ]:




