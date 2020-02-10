"""Load Modules"""
import os
import cv2
from glob import glob
import numpy as np
import random
import tensorflow as tf
from tqdm import tqdm

tf.enable_eager_execution()
from imageio import imread, imsave
import tensorflow.contrib.slim as slim

##
import matplotlib.pyplot as plt
%matplotlib notebook

from dh_segment.io import PAGE
from dh_segment.inference import LoadedModel
from dh_segment.post_processing import boxes_detection, binarization

from sklearn.manifold import TSNE


#########
# utils #
#########
def _signature_def_to_tensors(signature_def):
    g = tf.get_default_graph()
    return {k: g.get_tensor_by_name(v.name) for k, v in signature_def.inputs.items()}, \
           {k: g.get_tensor_by_name(v.name) for k, v in signature_def.outputs.items()}



"""Set gpu device"""
os.environ["CUDA_VISIBLE_DEVICES"]="0"

"""Start session"""
sess = tf.InteractiveSession()

"""Load model"""
model_dir = 'models/ENP_500_model_v3/export/1564890842'
m = LoadedModel(model_dir, predict_mode='filename')

"""Localize latent space"""
dhSegment_graph = tf.get_default_graph()
for op in dhSegment_graph.get_operations():
    print(op.values())

latent_space = dhSegment_graph.get_tensor_by_name('resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0')
print(latent_space)

loaded_model = tf.saved_model.loader.load(sess, ['serve'], model_dir)
list(loaded_model.signature_def)
input_dict_key = 'filename'
signature_def_key = 'serving_default'
input_dict, output_dict = _signature_def_to_tensors(loaded_model.signature_def[signature_def_key])
output_dict.update( {'latent' : latent_space} )
print(input_dict)
print(output_dict)


"""Prepare dataset"""
image_lists = glob('PATH/TO/ENP DATASET/val/images/*.jpg')
print("Total {} images".format(len(image_lists)))

"""Collect deep representation"""
deep_reps = []
for image_list in tqdm(image_lists):
    output = sess.run(output_dict, feed_dict={input_dict[input_dict_key]: image_list})
    deep_rep = np.mean(output['latent'][0],axis=(0,1))
    deep_reps.append(deep_rep)

"""Cluster deep representation"""
X_2d = TSNE(n_components=3).fit_transform(deep_reps)

"""Visualize clusters"""
plt.figure(figsize=(50, 50))
for i in range(len(image_lists)):
    plt.scatter(X_2d[i, 0], X_2d[i, 1])
#plt.show()