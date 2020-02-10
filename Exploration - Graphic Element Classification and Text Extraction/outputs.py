#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import mxnet
from mxnet.gluon.nn import HybridBlock, HybridSequential, Dense, GlobalMaxPool2D, Conv2D
import pooling


# In[ ]:


class DIQA_attach(HybridBlock):
    def __init__(self, nb_cls, **kwargs):
        super(DIQA_attach, self).__init__(**kwargs)
        
        self.maxpool = GlobalMaxPool2D()
        self.minpool = pooling.GlobalMinPool2D()
        self.fc1 = Dense(1024)
        self.fc2 = Dense(1024)
        self.out = Dense(nb_cls)
        
    def hybrid_forward(self, F, x):
        
        maxpool = self.maxpool(x)
        minpool = self.minpool(x)
        cat = F.Concat(maxpool, minpool)
        
        x = self.fc1(cat)
        x = self.fc2(x)
        x = self.out(x)
        x = F.softmax(x, axis=0)
        return x
    
    def __repr__(self):
        return self.__class__.__name__


# In[ ]:


class simple_attach(HybridBlock):
    def __init__(self, nb_cls, **kwargs):
        super(simple_attach, self).__init__(**kwargs)

        self.fc1 = Dense(4096)
        self.fc2 = Dense(1024)
        
        self.out = Dense(nb_cls)
        
    def hybrid_forward(self, F, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        x = F.softmax(x)
#         x = F.argmax(x)
        return x
    
    def __repr__(self):
        return self.__class__.__name__


# In[ ]:


class multilabel_attach(HybridBlock):
    def __init__(self, nb_cls, **kwargs):
        super(multilabel_attach, self).__init__(**kwargs)

        self.nb_cls = nb_cls
        self.conv1 = Conv2D(nb_cls, kernel_size=1, use_bias=False, prefix='tail')
        self.fc1 = Dense(1024, flatten=False)
        self.fc2 = Dense(1024, flatten=False)
        self.out = Dense(1, flatten=False)
        
    def hybrid_forward(self, F, x):
        
        x = self.conv1(x)
        x = x.reshape((0, self.nb_cls, -1))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        x = F.sigmoid(x)
#         x = F.argmax(x)
        return x
    
    def __repr__(self):
        return self.__class__.__name__

