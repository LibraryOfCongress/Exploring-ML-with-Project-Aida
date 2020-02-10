"""Load Modules"""
import os
import cv2
from glob import glob
import numpy as np
import random
import tensorflow as tf
from imageio import imread, imsave
import tensorflow.contrib.slim as slim

import matplotlib.pyplot as plt
%matplotlib notebook

from dh_segment.io import PAGE
from dh_segment.inference import LoadedModel
from dh_segment.post_processing import boxes_detection, binarization


os.environ["CUDA_VISIBLE_DEVICES"]="0"

"""Start session"""
session = tf.InteractiveSession()

"""Load model"""
# Path for model
model_dir = 'models/ENP_500_model_v3/export/1564890842/'

m = LoadedModel(model_dir, predict_mode='filename')

"""Download a single test image using URL to local server"""

# Download image
!wget -O 'my_test_image.jp2' -A jpeg,jpg,jp2 $TEST_IMAGE_URL
# Convert format from jp2 to jpg
img = cv2.imread('./my_test_image.jp2',cv2.IMREAD_GRAYSCALE)
cv2.imwrite('./images/my_test_image.jpg',img)
# Remove jp2
!rm my_test_image.jp2

"""Load a test image"""
file_to_process = './dataset/ENP_500/train/images/00674399.jpg'
img = cv2.imread(file_to_process)

"""Run prediction"""
prediction_outputs = m.predict(file_to_process)

"""POST PROCESSING"""
# Generate polygon regions
CONNECTIVITY = 8
THRESHOLD_SM_ZONE = 200*200

# Prepare img_copy for drawing Bounding-box
img_bb = np.copy(img)
img_pg = np.copy(img)
newH,newW = np.shape(img)[:2]

# Build figure mask from prediction
pred_figures = np.copy(prediction_outputs['labels'][0]).astype(np.uint8)
prob_figures = np.copy(prediction_outputs['probs'][0][:,:,2])
mask_figures = np.copy(pred_figures)
mask_figures[mask_figures != 2] = 0
mask_figures = (mask_figures//2)

# Resize mask to original image size
prob_figures = cv2.resize(prob_figures,(int(newW),int(newH)))
pred_figures = cv2.resize(pred_figures,(int(newW),int(newH)))
mask_figures = cv2.resize(mask_figures,(int(newW),int(newH)))


"""OPTION 1: Bounding-box"""
# Connected component analysis
fi_num_labels, fi_labels, fi_stats, fi_centroids = cv2.connectedComponentsWithStats(mask_figures, CONNECTIVITY, cv2.CV_32S)

# Draw Bounding-boxes
for i in range(fi_num_labels):
    # Ignore small zone
    if fi_stats[i,4] < THRESHOLD_SM_ZONE:
        continue
    c,r,w,h = fi_stats[i,0:4]
    cv2.rectangle(img_bb,(c,r),(c+w,r+h),(255,165,0),50)

"""OPTION 2: Polygon"""
# Connected component analysis
contours,_ = cv2.findContours(mask_figures, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Draw Polygon
for contour in contours:
    if cv2.contourArea(contour) < THRESHOLD_SM_ZONE:
        continue
    cv2.drawContours(img_pg, contour, -1, (255, 0, 0), 50)

    
fig = plt.figure(figsize=(2,2))
fig_ = fig.add_subplot(2,2,1)
fig_.title.set_text("Input")
fig_.imshow(img)
fig_ = fig.add_subplot(2,2,2)
fig_.title.set_text("Probability Map")
fig_.imshow(prob_figures)
fig_ = fig.add_subplot(2,2,3)
fig_.title.set_text("[Option 1] Output with boundingbox")
fig_.imshow(img_bb)
fig_ = fig.add_subplot(2,2,4)
fig_.title.set_text("[Option 2] Output with ploygon")
fig_.imshow(img_pg)

