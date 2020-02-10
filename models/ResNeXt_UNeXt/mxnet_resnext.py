#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# MXnet implementation of ReNeXt
#     reference: Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2017). 
#                Aggregated residual transformations for deep neural networks. 
#                In Proc. of IEEE conf. on CVPR


# In[ ]:


from __future__ import division

__all__ = ['ResNeXt', 
           'AggregatedBottleneck', 'BlockRepeater', 
           'resnext101_32x4d', 'resnext101_32x48d']

import os

import mxnet as mx
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn


# In[ ]:


class BlockRepeater(nn.HybridBlock):
    r""" Repeat residual blocks based on configuration
    
    Parameters
    ----------
    num_block : int
        Number of repeated block
    channel : int
        Number of channels
    cardianlity : int
        Number of column
    block_type : string, default 'AggregatedBottleneck'
        Block type being repeating
    is1stblock : bool, default False
        is the repeating block the first residual block
    """
    def __init__(self, num_block, channel, cardianlity, stage_num, block_type = 'AggregatedBottleneck', is1stblock = False, **kwargs):
        super(BlockRepeater, self).__init__(**kwargs)
        if (block_type == 'AggregatedBottleneck'):
            block = AggregatedBottleneck
        self.blocks = nn.HybridSequential(prefix='stage' + stage_num + '_')
        with self.blocks.name_scope():
            if (is1stblock):
                self.blocks.add(block(channels=channel, 
                                      stride=1, 
                                      cardianlity=cardianlity, 
                                      unit_num = '1',
                                      isDownsample=True))
                #print('0-0')
            else:
                self.blocks.add(block(channels=channel, 
                                      stride=2, 
                                      cardianlity=cardianlity, 
                                      unit_num = '1',
                                      isDownsample=True))
                #print('0-1')
            for c in range(num_block-1):
                self.blocks.add(block(channels=channel, 
                                      stride=1, 
                                      cardianlity=cardianlity, 
                                      unit_num = str(c + 2),
                                      isDownsample=False))
                #print(j)
    
    def hybrid_forward(self, F, x):
        return self.blocks(x)


# In[ ]:


class AggregatedBottleneck(nn.HybridBlock):
    r"""Aggregated Bottleneck from "Aggregated residual transformations for deep neural networks"
    
    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    cardianlity : int
        Number of column
    downsample : bool, default False
        Whether to downsample the input.
    identity_shortcut : bool, default False 
        Implement if needed - 
            if identity layer used, 
            then, reshape if input plane != output plane and use identity for shortcut
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, cardianlity, unit_num, isDownsample=False, identity_shortcut=False, in_channels=0, **kwargs):
        super(AggregatedBottleneck, self).__init__(**kwargs)
        #self.col_width = channels // (cardianlity * 2)
        #self.cardianlity = cardianlity
        #print(channels)
        self.group = nn.HybridSequential(prefix='unit' + unit_num + '_')
        self.group.add(nn.Conv2D(channels, kernel_size=1, strides=(1,1), use_bias=False))
        self.group.add(nn.BatchNorm())
        self.group.add(nn.Activation('relu'))
        self.group.add(nn.Conv2D(channels, kernel_size=3, strides=stride,padding=1, groups=cardianlity, use_bias=False))
        self.group.add(nn.BatchNorm())
        self.group.add(nn.Activation('relu'))
        self.group.add(nn.Conv2D(channels, kernel_size=1, strides=(1,1), use_bias=False))
        self.group.add(nn.BatchNorm())
        
        #self.group = []
        #for i in range(cardianlity):
        #    #print(i)
        #    item = nn.HybridSequential()
        #    item.add(nn.Conv2D(self.col_width, kernel_size=1, strides=(1,1)))
        #    item.add(nn.BatchNorm())
        #    item.add(nn.Activation('relu'))
        #    item.add(nn.Conv2D(self.col_width, kernel_size=3, strides=stride,padding=1))
        #    item.add(nn.BatchNorm())
        #    item.add(nn.Activation('relu'))
        #    item.add(nn.Conv2D(channels=self.col_width*2, kernel_size=1, strides=(1,1)))
        #    item.add(nn.BatchNorm())
        #    self.register_child(item)
        #    self.group.append(item)
        
        if isDownsample:
            self.sc = nn.Conv2D(channels, kernel_size=1, strides=stride,
                                use_bias=False, in_channels=in_channels)
            self.sc_bn = nn.BatchNorm()
        else:
            self.sc = None
        
    def hybrid_forward(self, F, x):
        residual = x
        #branch = None
        #for i in range(1,self.cardianlity - 1):
        #    if i == 1:
        #        branch = F.concat(self.group[i-1](x),self.group[i](x))
        #    else:
        #        branch = F.concat(branch,self.group[i](x))
        #x = F.concat(branch,self.group[self.cardianlity-1](x))

        #x = F.concat(self.group[0](x),self.group[1](x),self.group[2](x),self.group[3](x),
        #            self.group[4](x),self.group[5](x),self.group[6](x),self.group[7](x),
        #            self.group[8](x),self.group[9](x),self.group[10](x),self.group[11](x),
        #            self.group[12](x),self.group[13](x),self.group[14](x),self.group[15](x),
        #            self.group[16](x),self.group[17](x),self.group[18](x),self.group[19](x),
        #            self.group[20](x),self.group[21](x),self.group[22](x),self.group[23](x),
        #            self.group[24](x),self.group[25](x),self.group[26](x),self.group[27](x),
        #            self.group[28](x),self.group[29](x),self.group[30](x),self.group[31](x),)
        x = self.group(x)
        if self.sc:
            residual = self.sc_bn(self.sc(residual))
        x = F.Activation(residual+x, act_type='relu')
        return x


# In[ ]:


def ResNeXt(_cfgs, _cardianlity, _basewidth, _out_class):
    if (len(_cfgs) != 4):
        print("Layer config must be a vector of 4 values.")
        return None
        
    if (not isinstance(_cardianlity, int)):
        print("cardianlity must be a integer number.")
        return None
    
    if (not isinstance(_basewidth, int)):
        print("basewidth must be a integer number.")
        return None
    
    _ch = _cardianlity * _basewidth
    _model = nn.HybridSequential()
    _model.add(nn.BatchNorm())
    _model.add(nn.Conv2D(channels=64, kernel_size=7, strides=2, padding=3, use_bias=False))
    _model.add(nn.BatchNorm())
    _model.add(nn.MaxPool2D(pool_size=3, strides=(2,2), padding=1))
    
    for i, cfg in enumerate(_cfgs):
        #print(i, cfg)
        if (not isinstance(cfg, int)):
            print("Layer confg must be integer number.")
            return None
        if (i == 0):
            _model.add(BlockRepeater(num_block=cfg, 
                                     channel=(_ch * (2**(i))), 
                                     cardianlity=_cardianlity, 
                                     stage_num = str(i + 1),
                                     is1stblock=True))
        else:
            _model.add(BlockRepeater(num_block=cfg, 
                                     channel=(_ch * (2**(i))), 
                                     cardianlity=_cardianlity, 
                                     stage_num = str(i + 1),
                                     is1stblock=False))

    _model.add(nn.GlobalAvgPool2D())
    _model.add(nn.Dense(_out_class))
    return _model


# In[ ]:


def resnext101_32x4d(num_classes):
    _cfg = [3, 4, 23, 3]
    _cardianlity = 32
    _basewidth = 4
    return ResNeXt(_cfg, _cardianlity, _basewidth, num_classes)

def resnext101_32x48d(num_classes):
    _cfg = [3, 4, 23, 3]
    _cardianlity = 32
    _basewidth = 48
    return ResNeXt(_cfg, _cardianlity, _basewidth, num_classes)


# In[ ]:


"""test
model = resnext101_32x4d(1000)

net = model
net.hybridize()
net.collect_params().initialize()
sx = mx.sym.var('data')
sym = net(sx)
graph = mx.viz.plot_network(sym)
graph.format = 'tif'
graph.render('model')

model.initialize()
import numpy as np
np.random.seed(1)
xs = []
for i in range(10):
    num_x = np.random.rand(3,224,224)
    xs.append(num_x)
x=mx.nd.array(xs)
x.shape
y=model(x)
model.summary(x)
print(model)
"""


# In[ ]:




