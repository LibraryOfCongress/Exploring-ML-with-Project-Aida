#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# MXnet implementation of ReNeXt-UNet
#     reference: Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2017). 
#                Aggregated residual transformations for deep neural networks. 
#                In Proc. of IEEE conf. on CVPR
#     reference: Ronneberger, O., Fischer, P., & Brox, T. (2015, October). 
#                U-net: Convolutional networks for biomedical image segmentation.
#                In Intl. Conf. on Medi. ima. comput. & comput.-assis. interv.  
#
# U-Net code is adopted from https://github.com/chinakook/U-Net
#
# Dependency:
#     [model/mxnet_resnext.py]

from __future__ import division

__all__ = ['UNeXt', 'BlockRepeater',
           'UNeXt101_32x4d', 'UNeXt101_32x48d', 'UNeXt101_64x4d']

import os

import mxnet as mx
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn

import mxnet_resnext as res


# In[ ]:


def ConvBlock(channels, kernel_size):
    out = nn.HybridSequential()
    #with out.name_scope():
    out.add(
        nn.Conv2D(channels, kernel_size, padding=kernel_size // 2, use_bias=False),
        nn.BatchNorm(),
        nn.Activation('relu')
    )
    return out

def down_block(channels):
    out = nn.HybridSequential()
    #with out.name_scope():
    out.add(
        ConvBlock(channels, 3),
        ConvBlock(channels, 3)
    )
    return out


class up_block(nn.HybridBlock):
    def __init__(self, channels, stage_num, repeat=1, shrink=True, **kwargs):
        super(up_block, self).__init__(**kwargs)
        #with self.name_scope():

        # self.upsampler = nn.Conv2DTranspose(channels=channels, kernel_size=4, strides=2, 
                                            # padding=1, use_bias=False, groups=channels, weight_initializer=mx.init.Bilinear())

        # self.upsampler.collect_params().setattr('gred_req', 'null')

        self.conv1 = ConvBlock(channels, 1)
        self.conv3_0 = ConvBlock(channels, 3)
        if shrink:
            self.conv3_1 = ConvBlock(channels // 2, 3)
        else:
            self.conv3_1 = ConvBlock(channels, 3)
            
        #self.conv1_1 = nn.Conv2D(channels, kernel_size=1, use_bias=False)
        # self.conv3_1 = nn.HybridSequential(prefix='up_stage' + stage_num + '_')
        # if shrink:
            # for c in range(repeat):
                # self.conv3_1.add(res.AggregatedBottleneck(channels=channels//2, 
                                                          # stride=1, 
                                                          # cardianlity=32, 
                                                          # unit_num = str(c + 1),
                                                          # isDownsample=False))

            # self.conv1 = nn.Conv2D(channels // 2, kernel_size=1, use_bias=False)
        # else:
            # for c in range(repeat):
                # self.conv3_1.add(res.AggregatedBottleneck(channels=channels, 
                                                          # stride=1, 
                                                          # cardianlity=32, 
                                                          # unit_num = str(c + 1),
                                                          # isDownsample=False))
                                                    
            # self.conv1 = nn.Conv2D(channels, kernel_size=1, use_bias=False)
            
    def hybrid_forward(self, F, x, s):

        # x = self.upsampler(x)
        #s = F.relu(s)
        x = F.UpSampling(x, scale=2, sample_type='nearest')
        x = self.conv1(x)
        #x = F.relu(x)

        #x = F.Crop(*[x,s], center_crop=True)
        x = F.concat(s,x, dim=1)
        #x = self.conv1_1(x)
        #x = s + x
        x = self.conv3_0(x)
        
        #x = self.conv1(x)
        x = self.conv3_1(x)
        
        return x


# In[ ]:


class UNeXt(nn.HybridBlock):
    r"""ResNeXt-UNet
    
    Parameters
    ----------
    _cfgs : 4 integers vector
        ResNeXt layer configuration
    cardianlity : int
        Number of column
    
    _out_plane :
    downsample : bool, default False
        Whether to downsample the input.
    identity_shortcut : bool, default False 
        Implement if needed - 
            if identity layer used, 
            then, reshape if input plane != output plane and use identity for shortcut
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, _cfgs, _cardianlity, _basewidth, **kwargs):
        super(UNeXt, self).__init__(**kwargs)

        _ch = _cardianlity * _basewidth

        with self.name_scope():
            self.d0 = nn.HybridSequential()
            self.d0.add(nn.BatchNorm())
            self.d0.add(nn.Conv2D(channels=64, kernel_size=7, strides=2, padding=3, use_bias=False))
            self.d0.add(nn.BatchNorm())
            
            self.max0 = nn.HybridSequential()
            self.max0.add(nn.MaxPool2D(pool_size=3, strides=(2,2), padding=1))

            cfg0 = _cfgs[0]
            self.d1 = nn.HybridSequential()
            self.d1.add(res.BlockRepeater(num_block=cfg0, 
                                          channel=_ch, 
                                          cardianlity=_cardianlity,
                                          stage_num = '1',
                                          is1stblock=True))
            #self.d1.add(nn.Activation('relu'))
            cfg1 = _cfgs[1]
            self.d2 = nn.HybridSequential()
            self.d2.add(res.BlockRepeater(num_block=cfg1, 
                                          channel=_ch * 2, 
                                          cardianlity=_cardianlity, 
                                          stage_num = '2',
                                          is1stblock=False))
            #self.d2.add(nn.Activation('relu'))
            cfg2 = _cfgs[2]
            self.d3 = nn.HybridSequential()
            self.d3.add(res.BlockRepeater(num_block=cfg2, 
                                          channel=_ch * 4, 
                                          cardianlity=_cardianlity, 
                                          stage_num = '3',
                                          is1stblock=False))
            #self.d3.add(nn.Activation('relu'))
            #self.d3_512 = nn.Conv2D(channels=512, kernel_size=1, use_bias=True)
            cfg3 = _cfgs[3]
            self.d4 = nn.HybridSequential()
            self.d4.add(res.BlockRepeater(num_block=cfg3, 
                                          channel=_ch * 8, 
                                          cardianlity=_cardianlity, 
                                          stage_num = '4',
                                          is1stblock=False))
            #self.d4.add(nn.Activation('relu'))
            #self.d4.add(nn.Conv2D(channels=512, kernel_size=1, use_bias=True))
            self.u3 = up_block(_ch*4, '1', _cfgs[3], shrink=True)
            self.u2 = up_block(_ch*2, '2', _cfgs[2], shrink=True)
            self.u1 = up_block(_ch, '3', _cfgs[1], shrink=True)
            self.u0 = up_block(_ch//2, '4', _cfgs[0], shrink=True)
            self.uc = up_block(_ch//4, '5', 1, shrink=True)
            
            self.conv = nn.HybridSequential()
            self.conv.add(nn.Conv2D(6,1))

    def hybrid_forward(self, F, x):

        xc = x #224

        x0 = self.d0(x) # 1st conv
        x1 = self.max0(x0)
        x1 = self.d1(x1) # stage 1
        x2 = self.d2(x1) # stage 2
        x3 = self.d3(x2) # stage 3
        x4 = self.d4(x3) # stage 4
        #x3 = self.d3_512(x3)
        y3 = self.u3(x4,x3)
        y2 = self.u2(y3,x2)
        y1 = self.u1(y2,x1)
        y0 = self.u0(y1,x0)
        yc = self.uc(y0,xc)
        #yc = F.relu(yc)
        
        out = self.conv(yc)

        return out


# In[ ]:


def unext101_32x4d():
    _cfg = [3, 4, 23, 3]
    _cardianlity = 32
    _basewidth = 4
    _model = nn.HybridSequential(prefix='unext101_32x4d_')
    _model.add(UNeXt(_cfg, _cardianlity, _basewidth))
    return _model
def unext101_32x48d():
    _cfg = [3, 4, 23, 3]
    _cardianlity = 32
    _basewidth = 48
    _model = nn.HybridSequential(prefix='unext101_32x48d_')
    _model.add(UNeXt(_cfg, _cardianlity, _basewidth))
    return _model
def unext101_64x4d():
    _cfg = [3, 4, 23, 3]
    _cardianlity = 64
    _basewidth = 4
    _model = nn.HybridSequential(prefix='unext101_64x4d_')
    _model.add(UNeXt(_cfg, _cardianlity, _basewidth))
    return _model


# In[ ]:


"""test
model = unext101_64x4d()

net = model
net.hybridize()
net.collect_params().initialize()
sx = mx.sym.var('data')
sym = net(sx)
graph = mx.viz.plot_network(sym)
graph.format = 'tif'
graph.render('model_unext')

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




