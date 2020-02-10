# Introduction

"Exploration - Document Segmentation" aims to accomplish the following two objectives:
* Segmentation - Segment a document image into 5 different types of regions (background/text/figure/line-break/table)
* Clustering - Cluster document images based on visual similiarity

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The required software systems and libraries are:
* Anaconda >= 4.3
* Python >= 3.6
* TensorFlow 1.13
* CUDA 10.0 [if training on GPU]
* imageio >= 2.5
* pandas >= 0.24.2
* shapely >= 1.6.4
* scikit-learn >= 0.20.3
* scikit-image >= 0.15.0
* opencv-python >= 4.0.1
* tqdm >= 4.31.1
* sacred 0.7.4
* requests >= 2.21.0
* click >= 7.0

### Installing

A step by step instruction on how to install required software systems and libraries.

1. Download Python 3.6 from <https://www.python.org/downloads/>
2. Download CUDA 10.0 from <https://developer.nvidia.com/cuda-toolkit-archive>
3. Install Anaconda or Miniconda ([installation procedure](https://conda.io/docs/user-guide/install/index.html#))
4. Open Terminal (for MacOS), Command-Line (for Windows)
5. Go to the codebase/Exploration - Document Segmentation folder
6. Create a virtual environment and activate it
```
conda create -n segmentation python=3.6
source activate segmentation
```
7. Install packages
```
python setup.py install
```


## Data Acquisition
### Beyond Words
#### Collecting images
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

##### 3. Create `classes.txt` file
This file specifies overall class information. By default, please copy the contents below and save it under the root of the user-defined local directory.
```
0 0 0
255 0 0
0 255 0
0 0 255
255 255 0
255 0 255
```
Note: This scheme has to align with the values (line 137-141) in `BW2MulChanGT.py`.

The final expected directory structure for the dataset is elaborated below:
```
.Beyond_Words
├── train          # dataset for training
│   ├── images  
│   │   ├── npnp-jpeg-..._0089.jpg     # image_id.jpg
│   │   ├── ...
│   │   └── npnp-jpeg-..._0089.jpg  
│   └── labels
│       ├── npnp-jpeg-..._0159.png     # image_id.png
│       ├── ...
│       └── npnp-jpeg-..._0159.png  
├── val            # dataset for validation
│   ├── images  
│   └── labels
└── classes.txt
```


### Europeana Newspapers Project
#### Collecting images
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

##### 3. Create `classes.txt` file
This file specifies overall class information. By default, please copy the contents below and save it under the root of the user-defined local directory.
```
0 0 0 bg
255 0 0 txt
0 255 0 fig
0 0 255 line break
255 255 0 tb
```
Note: This scheme has to align with the values (line 143-154) in `ENP2MulChanGT.py`.

The final expected directory structure for the dataset is elaborated below:
```
.ENP_500
├── train          # dataset for training
│   ├── images  
│   │   ├── 00008061.jpg     # image_id.jpg
│   │   ├── ...
│   │   └── 00680328.jpg
│   └── labels
│       ├── 00008061.png     # image_id.png
│       ├── ...
│       └── 00680328.png
├── val            # dataset for validation
│   ├── images  
│   └── labels
└── classes.txt
```

## Running the training
1. Prepare the dataset as described above under the 'Exploration - Document Segmentation' folder
2. Actiavte the virtual environment and run the training script
```
source activate segmentation
python train.py with ENP_500_config_v3.json
```

## Running demo
1. Run the script, `demo_segmentation.py` provided in [demo/Exploration - Document Segmentation](https://git.unl.edu/unl_loc_summer_collab/codebase/tree/master/demo/Exploration%20-%20Document%20Segmentation)
2. Replace `TEST_IMAGE_URL` with your testing image url at line 33 in `demo_segmentation.py` and save it
3. Run one of the following command, depending on the purpose
```
# Activate virtual environment
source activate segmentation
# For segmentation task
python demo_segmentation.py
# For clustering task
python demo_clustering.py
```

## Built With

* [Python](https://www.python.org/) - The programming language
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) - Enable GPU for model training
* [TensorFlow](https://www.tensorflow.org/) - Deep learning framework

## Contributing

Segmentation - Enrich metadata generation by providing a set of item-level conceptual metadata<br/>
Clustering - Clustered manifold based on visual similarity can help page-level item retrieval (e.g., visually similar image suggestion)

## Authors
* **Benoit Seguin** and **Sofia Ares Oliveira** - DHLAB, EPFL - *git* - https://github.com/dhlab-epfl/dhSegment
* **Chulwoo Pack** - University of Nebraska-Lincoln - *email* - cpack@cse.unl.edu

