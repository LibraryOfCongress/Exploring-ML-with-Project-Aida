# Introduction

"Exploration - Document Type Classification" aims to build a deep learning model capable of distinguishing three different types of documents: (1) handwritten, (2) typed/machine-printed, and (3) mixed (both handwritten and typed). To this end, we propose to use a VGG-16 pre-trained on [RVL_CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) (400,000 labeled grayscale document images from 16 classes) for the task of document image classification.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The required software systems and libraries are:
* Anaconda >= 4.3
* Python >= 3.6
* TensorFlow 1.13
* CUDA 10.0 [if training on GPU]
* opencv-python >= 4.0.1
* numpy >= 1.16.2
* scikit-learn >= 0.20.3
* scikit-image >= 0.15.0
* matplotlib >= 1.4.3
* pandas >= 0.24.2
* seaborn 0.9.0

### Installing

A step-by-step instruction on how to install required software systems and libraries.

1. Download Python 3.6 from <https://www.python.org/downloads/>
2. Download CUDA 10.0 from <https://developer.nvidia.com/cuda-toolkit-archive>
3. Install Anaconda or Miniconda ([installation procedure](https://conda.io/docs/user-guide/install/index.html#))
4. Open Terminal (for MacOS), Command-Line (for Windows)
5. Go to the codebase/project1 folder
6. Create a virtual environment and activate it
```
conda create -n classification python=3.6
source activate classification
```
7. Install packages
```
python setup.py install
```

## Data Acquisition
As a dataset, we will download a set of document images from the Library of Congress Suffrage collections using our custom downloader script, which is based on [loc.gov JSON API](https://libraryofcongress.github.io/data-exploration/).
```
python LoC_Collection_Downloader.py
```
Our custom script downloads and saves images and label information (`labels.txt`) under the user-defined local directory, which will follow the structure shown below:
```
.suffrage_1002
├── labels.txt                # label info for each image; 0: handwritten, 1: typed, 2: mixed
└── images
    ├── service_mss_mss41210_mss41210-002_00049_6.jpg      # image_id.jpg
    ├── ...      
    └── service_mss_mss11049_mss11049-001_00013_3.jpg
```
Note that the list of image URLs (`image_url_with_gt.xlsx`) had manually compiled and balanced across handwritten, typed/typeset, and mixed by members of the project team.

## Running the training and evaluation

1. Prepare the dataset and locate it under the 'Exploration - Document Type Classification' folder
2. Run the training script
```
python train.py
```
3. Run the evaluation script
```
python eval.py
```

## Built With

* [Python](https://www.python.org/) - The programming language
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) - Enable GPU for model training
* [TensorFlow](https://www.tensorflow.org/) - Deep learning framework

## Contributing

Classification - Enrich metadata generation by providing page-level document type metadata

## Authors
* **Chulwoo Pack** - University of Nebraska-Lincoln - *email* - cpack@cse.unl.edu