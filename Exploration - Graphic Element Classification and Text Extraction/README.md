# Introduction

Exploration - Graphic Element Classification and Text Extraction finds and localizes Figure/Illustration/Cartoon presented in an image.

Note: to extract texts,
1. Take output of the model as the mask to segment the image
2. Feed segmented snipptes to [EAST](http://east.zxytim.com/)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The required software systems and libraries are:
* Python 3.7
* MXNet 1.5
* CUDA 10.0 [if training on GPU]
* Matplotlib 3.1.1
* opencv-python 4.1
* numpy 1.17

### Installing

A step-by-step instruction how to install the required software system and libraries.

1. Download Python 3.7 from <https://www.python.org/downloads/>
2. Download CUDA 10.0 from <https://developer.nvidia.com/cuda-toolkit-archive>
3. Install downloaded installation file
4. Open Terminal (for macOS), Command-Line (for Windows)
5. Install MXNet
```
pip install 'mxnet-cu100==1.5.1'
```
6. Install Matplotlib
```
python -m pip install -U 'matplotlib==3.1.1'
```
7. Install opencv-python
```
pip install 'opencv-python==4.1'
```
8. Install numpy
```
pip install 'numpy==1.17'
```
## Data Acquisition

### Collecting images from ENP dataset
As a dataset, we will download a subset of document images from the [Europeana Newspapers Project](https://www.primaresearch.org/datasets/ENP).
Note that we sample 481 pages by the following procedure:
1. Go to the [ENP database](https://www.primaresearch.org/datasets/ENP) (Registration is required to access)
2. Click **Browse Images** on the left
3. Click **All** under **Document Types** tap on the left
4. Select **Newspaper** and click **Apply**
5. Click **Add all to cart**
6. Move on to the next page and add more image to your cart as much as you want
7. Click **Your cart** on the left
8. Select **Original Images + Attachments** under  **Download contents** on the left
9. Click **Download**

#### Polishing dataset
Once images are downloaded, the following three steps are expected:
##### 1. Generate ground-truth for each image 
By running our script, ground-truth for each document image is generated and saved under the user-defined local directory.
```
python ENP2MulChanGT.py
```
A ground-truth is a 3-channel image where pixels in the same class share the same value.

##### 2. Separate images and ground-truth into two sub-directories: `train` and `val`
This process is automated in step 1, in which a user can adjust the separation ratio at line 21 in `ENP2MulChanGT.py`

##### 3. Create ground-truth file for each image
1. Open terminal
2. Create necessary folder
```
mkdir ENP_500/train/lbls
```
3. Copy the matlab script
```
cp rgb2lbl_bw.m ENP_500/train/rgb2lbl_bw.m
```
3. Execute the matlab script
```
matlab -nodisplay -r ENP_500/train/rgb2lbl_bw.m
```

### Collecting images from the Beyond Words dataset
As a dataset, we will download document images from the [Beyond Words](http://beyondwords.labs.loc.gov).
Note that we download document images by the following procedure:
1. Go to the [Beyond Words](http://beyondwords.labs.loc.gov/#/data)
2. Click **Download**
3. Right-click the page and save as a file named `beyondwords_gt.json`
4. Execute the python script `BW_Downloader.py` to download images
```
python BW_Downloader.py
```

Note that, for the purpose of reproduction of the result in the report. Please use the `beyondwords_gt.json` file provided in the `dataset` folder.

#### Polishing dataset
Once images are downloaded, the following three steps are expected:
##### 1. Generate ground-truth for each image 
By running our script, ground-truth for each document image is generated and saved under the user-defined local directory.
```
python BW2MulChanGT.py
```
A ground-truth is a 3-channel image where pixels in the same class share the same value.

##### 2. Separate images and ground-truth into two sub-directories: `train` and `val`
This process is automated in step 1, in which a user can adjust the separation ratio at line 18 in `BW2MulChanGT.py`

##### 3. Create ground-truth file for each image
1. Open terminal
2. Create necessary folder
```
mkdir Beyond_Words/train/lbls
```
3. Copy the matlab script
```
cp rgb2lbl_bw.m Beyond_Words/train/rgb2lbl_bw.m
```
3. Execute the matlab script
```
matlab -nodisplay -r Beyond_Words/train/rgb2lbl_bw.m
```

## Running the training process

1. Configurate the training
    1.1 set task type: (1) six-class classification or, (2), two-class classification
    1.2 set upscaling technique: (1), Deconvolutional layer or, (2), Resizing layer
    1.3 set the path to save the training log
    1.4 set if save the intermediate predictions
2. Download dataset from 
<https://git.unl.edu/unl_loc_summer_collab/labeled_data/tree/master/BeyondWord_orginal_resolution>
3. Copy the downloaded folder to the downloaded 'project2' folder
4. Run the training script
```
python train.py
```

### Formats of the training log

There are four files generated by the training process.
Training performance for each image in the training set.
```
seg_iou_bw_direct_batch_stat.txt
```
Testing performance for each image in the testing set.
```
seg_iou_bw_direct_test_batch_stat.txt
```
Each line of image-specific log consists of six parts split by "|".
They are:
1. the number ID of the input image;
2. the loss to the groundtruth for the input image;
3. the class-wise accuracy;
4. the average accuracy over all classes;
5. the class-wise mIoU;
6. the average mIoU over all classes.

Training performance for each training step.
```
seg_iou_bw_direct_train_epoch_stat.txt
```
Testing performance for each training step.
```
seg_iou_bw_direct_test_epoch_stat.txt
```
Each line of training-step-wise log consists of seven parts split by "|".
They are:
1. the number ID of the training step;
2. the time elapsed for the training step;
3. the average loss to the groundtruth for the training step;
4. the average class-wise accuracy over the training step;
5. the average accuracy over all classes for the training step;
6. the class-wise mIoU over the training step;
7. the average mIoU over all classes for the training step.

## Built With

* [Python](https://www.python.org/) - The programming language
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) - Enable GPU for model training
* [MXNet](https://mxnet.apache.org/) - Deep learning framework

## Contributing

An automated solution to extract figures/graphs is promising
Enrich page-level metadata by cataloging the types of visual components 
Enrich item-level metadata by extracting texts in figure/graph regions

## Authors

* **Yi Liu** - University of Nebraska-Lincoln - *email* - yil@cse.unl.edu