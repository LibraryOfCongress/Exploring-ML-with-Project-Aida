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

##########
# Params #
##########
# Path
DATA_ROOT = './dataset/suffrage_1002'
MODEL_CHECKPOINT_ROOT = './models'
LOG_ROOT = './logs'
# Model
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
MODEL_DESC = 'three_class_freeze_v3.0'
# Hyperparams
LEARNING_RATE = 0.0000005
SHUFFLE_BUFFER_SIZE = 64
BATCH_SIZE = 16
CLASS_IDS = [0,1,2]


#########
# Utils #
#########
# Use a custom OpenCV function to read the image, instead of the standard
# TensorFlow `tf.read_file()` operation.
def _read_py_function(filename,label):
    image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
    image_decoded = np.stack((image_decoded,)*3, axis=-1)
    return image_decoded.astype(np.uint8),label

# Use standard TensorFlow operations to resize the image to a fixed shape.
def _resize_function(image_decoded,label):
    image_decoded.set_shape([None, None, 3])
    label.set_shape([None,])
    image_resized = tf.image.resize_images(image_decoded, [IMG_SIZE, IMG_SIZE])
    image_resized /= 255.0
    return image_resized,label


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

# Split data per class
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

"""Build dataset for train"""
# image_ds
image_ds = tf.data.Dataset.from_tensor_slices((train_image_paths))
# label_ds
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_labels, tf.int64))
# image_label_ds
ds = tf.data.Dataset.zip((image_ds, label_ds))

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
ds = ds.shuffle(buffer_size=(int(len(train_image_paths))))

ds = ds.repeat()

ds = ds.map(
    lambda image_path,label: tuple(tf.py_func(_read_py_function, [image_path,label], [tf.uint8,label.dtype])))
ds = ds.map(_resize_function)

#label_ds = tf.data.Dataset.from_tensor_slices(np.expand_dims(np.array(train_labels),axis=1))
#aaa = np.random.sample((100,1))
#label_ds = tf.data.Dataset.from_tensor_slices(aaa)

ds = ds.batch(BATCH_SIZE)


# `prefetch` lets the dataset fetch batches in the background while the model is training.
ds = ds.prefetch(buffer_size=BATCH_SIZE)

ds_train = ds

print("Total train images:\t{}".format(len(train_image_paths)))
print("Total train labels:\t{}".format(len(train_labels)))

"""
Build dataset for validation
"""
# image_ds
image_ds = tf.data.Dataset.from_tensor_slices((val_image_paths))
# label_ds
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(val_labels, tf.int64))
# image_label_ds
ds = tf.data.Dataset.zip((image_ds, label_ds))
# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.

ds = ds.shuffle(buffer_size=(int(len(val_image_paths))))
ds = ds.repeat()

ds = ds.map(
    lambda image_path,label: tuple(tf.py_func(_read_py_function, [image_path,label], [tf.uint8,label.dtype])))
ds = ds.map(_resize_function)



ds = ds.batch(BATCH_SIZE)

# `prefetch` lets the dataset fetch batches in the background while the model is training.
ds = ds.prefetch(buffer_size=BATCH_SIZE)

ds_val = ds


print("Total val images:\t{}".format(len(val_image_paths)))
print("Total val labels:\t{}".format(len(val_labels)))

print(ds_val)


#########
# Model #
#########
base_model = tf.keras.applications.VGG16(input_shape=(224,224,3), \
                                         include_top=False, \
                                         weights='imagenet')
base_model.trainable=True


# [Option 1] Shallow
model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(len(CLASS_IDS), activation = 'softmax')])
model.load_weights(os.path.join(MODEL_CHECKPOINT_ROOT,'all_class_no_freeze_v2.0_56.h5'))
print("\n{} is loaded.".format(os.path.join(MODEL_CHECKPOINT_ROOT,'all_class_no_freeze_v2.0_56.h5')))


model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(len(CLASS_IDS), activation = 'softmax')
])

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate = LEARNING_RATE),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

model.summary()

# Callbacks

    
callbacks = [ 
    # [Option 1] Checkpoint callback
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_CHECKPOINT_ROOT,(MODEL_DESC+'_{epoch}.h5')),
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        save_best_only=True,
        monitor='val_loss',
        verbose=1),
    
    # [Option 2] Tensorboard callback
    tf.keras.callbacks.TensorBoard(log_dir=LOG_ROOT, \
                                   write_images=True),
    
    # [Option 3] Early stopping
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
]



#########
# Train #
#########
epochs = 200
history = model.fit(ds_train,
                    initial_epoch = 0,
                    steps_per_epoch = round(len(train_image_paths))//BATCH_SIZE,
                    validation_data=ds_val,
                    validation_steps = round(len(val_image_paths))//BATCH_SIZE,
                    epochs=epochs,
                    callbacks=callbacks)

################
# Save history #
################
with open(os.path.join(LOG_ROOT,(MODEL_DESC+'_train_200_HistDic')), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)