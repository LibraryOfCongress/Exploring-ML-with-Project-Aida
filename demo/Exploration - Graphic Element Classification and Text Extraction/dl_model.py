#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv

import modelzoo as zoo
import mxnet as mx
import mxnet_unext as unext

import warnings

import outputs


# In[ ]:


def load_pretrained_resnext101_64_4d(ctx,
                                     need_download = False):
    if need_download:
        zoo.download_model('imagenet1k-resnext-101-64x4d')
        
    symbol_path = 'imagenet1k-resnext-101-64x4d-symbol.json'
    param_path = 'imagenet1k-resnext-101-64x4d-0000.params'
    sym = mx.sym.load(symbol_path)
    inter = sym.get_internals()
#     print(inter)
    new_sym = inter['flatten0_output']

    inputsym = mx.sym.var('data', dtype=mx.base.mx_real_t)
    model = mx.gluon.nn.SymbolBlock(outputs=new_sym, inputs=inputsym)
    model.collect_params().load(param_path, ctx=ctx, allow_missing=False, ignore_extra=True)
    
    return model


# In[ ]:


def load_resnext101_64_4d(ctx,
                          out_sym,
                          need_download = False):
    if need_download:
        zoo.download_model('imagenet1k-resnext-101-64x4d')
        
    symbol_path = 'imagenet1k-resnext-101-64x4d-symbol.json'
    param_path = 'imagenet1k-resnext-101-64x4d-0000.params'
    sym = mx.sym.load(symbol_path)
    inter = sym.get_internals()
#     print(inter)
    new_sym = inter[out_sym + '_output']

    inputsym = mx.sym.var('data', dtype=mx.base.mx_real_t)
    model = mx.gluon.nn.SymbolBlock(outputs=new_sym, inputs=inputsym)
    model.collect_params().load(param_path, ctx=ctx, allow_missing=False, ignore_extra=True)
    
    return model


# In[ ]:


def load_pretrained_resnext_to_unext101_64_4d(ctx,
                                              need_download = False,
                                              fine_tune = False,
                                              migrate_input_norm = True,
                                              migration_list = 'layer_migration_list.csv'):

# Note: fine-tune here is wheather tune pre-trained part
    
    if need_download:
        zoo.download_model('imagenet1k-resnext-101-64x4d')

    unext_param_list = []
    resnext_params_list = []
    with open(migration_list, newline='') as csvfile:
        list_reader = csv.reader(csvfile, delimiter=',')
        for row in list_reader:
            unext_param_list.append(row[0])
            resnext_params_list.append(row[1])

    if not migrate_input_norm:
        unext_param_list = unext_param_list[4:len(unext_param_list)]
        resnext_params_list = resnext_params_list[4:len(resnext_params_list)]

    migration_dict = dict(zip(unext_param_list, resnext_params_list))
    #print(unext_param_list)
    #print(resnext_params_list)
    #print(migration_dict)

    
    #sym, arg_params, aux_params = mx.model.load_checkpoint('imagenet1k-resnext-101-64x4d', 0)
    model = unext.unext101_64x4d()
    model_params = model.collect_params()
    migration_params = mx.ndarray.load('imagenet1k-resnext-101-64x4d-0000.params')

    for key in migration_dict:
        model_params[key]._load_init(migration_params[migration_dict[key]],ctx)

    if not fine_tune:
        for param in unext_param_list:
            model_params[param].grad_req = 'null'
#         for param in model_params.values():
#             param.grad_req = 'null'

    return model


# In[ ]:


def resume_training_unext101_64_4d(ctx,
                                   symb,
                                   parame,
                                   need_download = False,
                                   fine_tune = False,
                                   freeze_input_norm = True,
                                   migration_list = 'layer_migration_list.csv'):

# Note: fine-tune here is wheather tune u-net part

#TODO     if need_download:
        #zoo.download_model('imagenet1k-resnext-101-64x4d')

    unext_param_list = []
    with open(migration_list, newline='') as csvfile:
        list_reader = csv.reader(csvfile, delimiter=',')
        for row in list_reader:
            unext_param_list.append(row[0])

    if not freeze_input_norm:
        unext_param_list = unext_param_list[4:len(unext_param_list)]

    #sym, arg_params, aux_params = mx.model.load_checkpoint('imagenet1k-resnext-101-64x4d', 0)
    model = mx.gluon.nn.SymbolBlock.imports(symb, ['data'], parame, ctx=ctx)
#     model.load_parameters('unext_strip-0000.params')
    model_params = model.collect_params()
    
    if fine_tune:
        for param in unext_param_list:
#             print(param)
            model_params[param].grad_req = 'null'

    else:
        for param in model_params:
            model_params[param].grad_req = 'null'
#         for param in model_params.values():
#             param.grad_req = 'null'

    return model

def eval_model(ctx, symb, param):
    model = mx.gluon.nn.SymbolBlock.imports(symb, ['data'], param, ctx=ctx)
#     model.load_parameters('unext_strip-0000.params')
    model_params = model.collect_params()
    for param in model_params:
        model_params[param].grad_req = 'null'
    return model

def resume_training_unext101_64_4d_beyond_word(ctx,
                                               symb,
                                               parame,
                                               need_download = False,
                                               fine_tune = False,
                                               freeze_input_norm = True,
                                               migration_list = 'layer_migration_list.csv'):

# Note: fine-tune here is wheather tune u-net part

#TODO     if need_download:
        #zoo.download_model('imagenet1k-resnext-101-64x4d')

    symbol_path = symb
    param_path = parame
    sym = mx.sym.load(symbol_path)
    inter = sym.get_internals()
#     print(inter)
    new_sym = inter['unext0_conv20_fwd_output']

    inputsym = mx.sym.var('data', dtype=mx.base.mx_real_t)
    model = mx.gluon.nn.SymbolBlock(outputs=new_sym, inputs=inputsym)
    model.collect_params().load(param_path, ctx=ctx, allow_missing=False, ignore_extra=True)
    
    unext_param_list = []
    with open(migration_list, newline='') as csvfile:
        list_reader = csv.reader(csvfile, delimiter=',')
        for row in list_reader:
            unext_param_list.append(row[0])

    if not freeze_input_norm:
        unext_param_list = unext_param_list[4:len(unext_param_list)]

    model_params = model.collect_params()
    
    if fine_tune:
        for param in unext_param_list:
#             print(param)
            model_params[param].grad_req = 'null'

    else:
        for param in model_params:
            model_params[param].grad_req = 'null'
#         for param in model_params.values():
#             param.grad_req = 'null'

    attachment = mx.gluon.nn.Conv2D(6,1)
    attachment.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    
    net = mx.gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(model)
        net.add(attachment)

    return net
# In[ ]:


def get_layer_output(symbol, arg_params, aux_params, layer_name):
    all_layers = symbol.get_internals()
    net = all_layers[layer_name+'_output']
#     net = mx.symbol.Flatten(data=net)
    new_args = dict({k:arg_params[k] for k in arg_params if k in net.list_arguments()})
    new_aux = dict({k:aux_params[k] for k in aux_params if k in net.list_arguments()})
    return (net, new_args, new_aux)


# In[ ]:


def fine_tune_subjective_diqa_unext101_64_4d(ctx,
                                             nb_cls,
                                             fine_tune = False,
                                             chkpt = None,
                                             symbol_path = None,
                                             param_path = None,
                                             need_download = False):
# Note: fine-tune here is wheather tune pre-trained part
#TODO     if need_download:
        #zoo.download_model('imagenet1k-resnext-101-64x4d')

#     model = None
    if chkpt == None:
            sym = mx.sym.load(symbol_path)
            inter = sym.get_internals()
        #     print(inter)
            new_sym = inter['unext0_conv19_fwd_output']

            inputsym = mx.sym.var('data', dtype=mx.base.mx_real_t)
            model = mx.gluon.nn.SymbolBlock(outputs=new_sym, inputs=inputsym)
            model.collect_params().load(param_path, ctx=ctx, allow_missing=False, ignore_extra=True)
        #     print(model.summary)
    elif symbol_path == None and param_path == None:
        sym, arg_params, aux_params = mx.model.load_checkpoint(chkpt, 0)
        new_sym, new_args, new_aux = get_layer_output(sym, arg_params, aux_params, 'unext0_conv19_fwd')
#         with warnings.catch_warnings():
        model = mx.gluon.nn.SymbolBlock(outputs=new_sym, inputs=mx.sym.var('data', dtype=mx.base.mx_real_t))
        net_params = model.collect_params()
        for param in new_args:
            if param in net_params:
                net_params[param]._load_init(new_args[param], ctx=ctx)
        for param in new_aux:
            if param in net_params:
                net_params[param]._load_init(new_aux[param], ctx=ctx)
    else:
        return None

    if not fine_tune:
        params = model.collect_params()
        for p in params:
            params[p].grad_req = 'null'
        
    attachment = outputs.DIQA_attach(nb_cls=5)
    attachment.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    
    net = mx.gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(model)
        net.add(attachment)
        
#     print(net.summary)

    return net


# In[ ]:


def fine_tune_resnext101_64_4d(ctx, 
                               nbCls,
                               fine_tune = False,
                               need_download = False):
# Note: fine-tune here is wheather tune pre-trained part
    model = load_pretrained_resnext101_64_4d(ctx=ctx, need_download=need_download)
    
    if not fine_tune:
        params = model.collect_params()
        for p in params:
            params[p].grad_req = 'null'
    
    attachment = outputs.simple_attach(nb_cls=nbCls)
    attachment.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    
    net = mx.gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(model)
        net.add(attachment)
        
#     print(net.summary)

    return net


# In[ ]:


def fine_tune_resnext101_64_4d_multilabel(ctx, 
                                          nbCls,
                                          fine_tune = False,
                                          need_download = False):
# Note: fine-tune here is wheather tune pre-trained part
    model = load_resnext101_64_4d(ctx=ctx, out_sym='pool1', need_download=need_download)
    
    if not fine_tune:
        params = model.collect_params()
        for p in params:
            params[p].grad_req = 'null'
    
    attachment = outputs.multilabel_attach(nb_cls=nbCls)
    attachment.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    
    net = mx.gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(model)
        net.add(attachment)
        
#     print(net.summary)

    return net


# In[ ]:


def resume_fine_tune_training(ctx,
                              resuming_model,
                              pretrain_params,
                              fine_tune = False,
                              need_download = False):
    
# Note: fine-tune here is wheather tune pre-trained part
# Note: this func does not work with u-next

#TODO need_download
    
    migration_params = mx.ndarray.load(pretrain_params)
    
    symbol_path = resuming_model+'-symbol.json'
    param_path = resuming_model+'-0000.params'
    model = mx.gluon.nn.SymbolBlock.imports(symbol_path, ['data'], param_path, ctx=ctx)

    
    if not fine_tune:
        model_params = model.collect_params()
        for param in model_params:
            if param in migration_params:
                model_params[param].grad_req = 'null'

    return model


# In[ ]:


"""test
_ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
#print(_ctx)
# model=load_pretrained_resnext101_64_4d(migrate_input_norm = False, fine_tune = False, ctx=_ctx)
# model=load_pretrained_resnext_to_unext101_64_4d(migrate_input_norm = False, fine_tune = False, ctx=_ctx)
# model=fine_tune_subjective_diqa_unext101_64_4d(ctx=_ctx, symbol_path="unext_10ep-symbol.json", param_path="unext_10ep-0000.params", nb_cls=13)
# model = load_pretrained_resnext101_64_4d(ctx = _ctx)
# model = fine_tune_resnext101_64_4d(ctx = _ctx, nbCls = 2)
model = fine_tune_resnext101_64_4d_multilabel(ctx = _ctx, nbCls = 13)

net = model
# print(net.get_internals())
net.hybridize()
# net.collect_params().initialize()

sx = mx.sym.var('data')
sym = net(sx)
graph = mx.viz.plot_network(sym)
graph.format = 'tif'
graph.render('dl_model')

with mx.Context(_ctx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.initialize(ctx=_ctx)
    import numpy as np
    np.random.seed(1)
    xs = []
    for i in range(1):
        num_x = np.random.rand(3,224,224)
        xs.append(num_x)
    x=mx.nd.array(xs)
    x.shape
    y=model(x)
    model.summary(x)
    print(model)
"""


# In[ ]:




