################
# Load modules #
################
# tensorflow
import tensorflow as tf
# basic
import cv2
import numpy as np
import math
import os
import pathlib
from glob import glob
import pickle
import datetime
# eval
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
import pandas as pd

##############
# Assign GPU #
##############
os.environ["CUDA_VISIBLE_DEVICES"]="0"



####################
# Data preparation #
####################
"""
Get image_paths and labels, and split into train, val, and test set.
""" 
# Read image path and labels
with open(os.path.join(DATA_ROOT,'labels.txt')) as _list:
    all_paths_with_labels = [(DATA_ROOT+'/'+line).rstrip('\n').split() for line in _list]

all_image_paths = np.array(all_paths_with_labels)[:,0].tolist()
all_labels      = (np.array(all_paths_with_labels)[:,1]).astype(int).tolist()

# Shuffle lists
_before_shuffle = list(zip(all_image_paths,all_labels))
random.seed(1)
random.shuffle(_before_shuffle)
all_image_paths,all_labels = zip(*_before_shuffle) 
all_image_paths,all_labels = list(all_image_paths),list(all_labels)

# Split data per class (80%,10%,10% for train, val, and test)
train_image_paths = []
train_labels = []
val_image_paths = []
val_labels = []
test_image_paths = []
test_labels = []
for class_id in CLASS_IDS:
    _image_paths = np.array(all_image_paths)[np.argwhere(np.array(all_labels)==class_id).squeeze().tolist()].tolist()
    _labels = (class_id*np.ones((len(_image_paths)))).astype(np.int64).tolist()
    train_image_paths += _image_paths[:int(0.5*0.8*len(_image_paths))]
    train_labels += _labels[:int(0.5*0.8*len(_image_paths))]
    val_image_paths += _image_paths[int(0.5*0.8*len(_image_paths)):int(0.5*0.8*len(_image_paths))+int(0.5*0.1*len(_image_paths))]
    val_labels += _labels[int(0.5*0.8*len(_image_paths)):int(0.5*0.8*len(_image_paths))+int(0.5*0.1*len(_image_paths))]
    test_image_paths += _image_paths[-int(0.5*0.1*len(_image_paths)):]
    test_labels += _labels[-int(0.5*0.1*len(_image_paths)):]
    
print("{} train images".format(len(train_image_paths)))
print("\tclass0:{}/class1:{}/class2:{}".format(len(np.argwhere(np.array(train_labels)==0)),len(np.argwhere(np.array(train_labels)==1)),len(np.argwhere(np.array(train_labels)==2))))
print("{} val images".format(len(val_image_paths)))
print("\tclass0:{}/class1:{}/class2:{}".format(len(np.argwhere(np.array(val_labels)==0)),len(np.argwhere(np.array(val_labels)==1)),len(np.argwhere(np.array(val_labels)==2))))
print("{} test images".format(len(test_image_paths)))
print("\tclass0:{}/class1:{}/class2:{}".format(len(np.argwhere(np.array(test_labels)==0)),len(np.argwhere(np.array(test_labels)==1)),len(np.argwhere(np.array(test_labels)==2))))


"""
Build dataset for test
"""
# image_ds
image_ds = tf.data.Dataset.from_tensor_slices((test_image_paths))
# label_ds
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(test_labels, tf.int64))
# image_label_ds
ds = tf.data.Dataset.zip((image_ds, label_ds))

ds = ds.map(
    lambda image_path,label: tuple(tf.py_func(_read_py_function, [image_path,label], [tf.uint8,label.dtype])))
ds = ds.map(_resize_function)

ds = ds.batch(BATCH_SIZE)
ds = ds.repeat()
# `prefetch` lets the dataset fetch batches in the background while the model is training.
ds = ds.prefetch(buffer_size=BATCH_SIZE)

ds_test = ds

print("Total test images:\t{}".format(len(test_image_paths)))
print("Total test labels:\t{}".format(len(test_labels)))



##############
# Evaluation #
##############
# Run evaluation for generating accuracy
steps = len(test_image_paths)//BATCH_SIZE
loss, acc = model.evaluate(ds_test,steps=steps+1)
print("Trained model, accuracy: {:5.2f}%".format(100*acc))

# Construct confusion matrix
y_true = np.array(test_labels)
y_pred = np.argmax(prediction,axis=-1)
conf_mat = confusion_matrix(y_true, y_pred)

# Construct dataframe; sewing data and labels
df_cm = pd.DataFrame(conf_mat, 
                        index = [i for i in LABELS],
                        columns = [i for i in LABELS])

# Visualize confusion matrix
"""
plt.figure(figsize=(5, 5))
fig = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
fig.set(xlabel="Prediction", ylabel="Actual")
plt.show(fig)
"""

precisions, recalls, f1s, supports = precision_recall_fscore_support(y_true, y_pred)