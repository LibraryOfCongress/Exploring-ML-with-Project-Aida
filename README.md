# Introduction

The University of Nebraska-Lincoln's (UNL) Aida digital libraries research team and the Library of Congress (LC) collaborated on a "summer of machine learning" in 2019  to explore machine learning techniques for extending the accessibility of digital collections. The UNL team developed a number of prototype explorations over multiple iterations to investigate a range of questions and issues related to the digital materials, the LC's collections, and to machine learning practices in cultural heritage organizations. The UNL team employed a variety of machine learning approaches such as back-propagation neural network-based classifiers and deep learning approaches, including  convolutional neural networks. More specifically, these projects involve VGG16, ResNeXt, dhSegment, and a fusion network combining ResNeXt and U-Net. 

This repository includes the code developed and used across the team's explorations.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

For Exploration - Document Segmentation, the required software systems and libraries are:
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

For Exploration - Graphic Element Classification and Text Extraction and Exploration - Digitization Type Differentiation, the required software systems and libraries are:
* Python 3.7
* MXNet 1.5
* CUDA 10.0 [if training on GPU]
* Matplotlib 3.1.1
* opencv-python 4.1
* numpy 1.17

For Exploration - Document Type Classification, the required software systems and libraries are:
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

For Exploration - Document Image Quality Assessment, the required software systems and libraries are:
* Python 3.7
* scipy 1.3.1
* opencv-python 4.1
* skimage 0.15

### Installing

Step-by-step instructions on how to install required software systems and libraries for each project

For Exploration - Document Segmentation

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

For Exploration - Graphic Element Classification and Text Extraction and Exploration - Digitization Type Differentiation

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

For Exploration - Document Type Classification

1. Download Python 3.6 from <https://www.python.org/downloads/>
2. Download CUDA 10.0 from <https://developer.nvidia.com/cuda-toolkit-archive>
3. Install Anaconda or Miniconda ([installation procedure](https://conda.io/docs/user-guide/install/index.html#))
4. Open Terminal (for MacOS), Command-Line (for Windows)
5. Go to the codebase/Exploration - Digitization Type Differentiation folder
6. Create a virtual environment and activate it
```
conda create -n classification python=3.6
source activate classification
```
7. Install packages
```
python setup.py install
```

For Exploration - Document Image Quality Assessment

1. Download Python 3.7 from <https://www.python.org/downloads/>
2. Install downloaded installation file
3. Open Terminal (for macOS), Command-Line (for Windows)
4. Install scipy
```
pip install 'scipy==1.3.1'
```
5. Install opencv-python
```
pip install 'opencv-python==4.1'
```
6. Install skimage
```
pip install 'scikit-image==0.15'
```

## Running the demonstrations

Exploration - Document Segmentation:
1. Download all files in demo/Exploration - Document Segmentation
<https://git.unl.edu/unl_loc_summer_collab/codebase/tree/master/demo/Exploration%20-%20Document%20Segmentation>
2. Install required softwares and libraries
3. Download one of the following dataset: 
(1) <https://git.unl.edu/unl_loc_summer_collab/labeled_data/tree/master/ENP_500> or 
(2) <https://git.unl.edu/unl_loc_summer_collab/labeled_data/tree/master/difficulty_collection>, 
for segmentation or clustering task, respectively
4. Copy the downlaoded folder to the downloaded 'Exploration - Document Segmentation' folder
5. Run one of the following command, depending on the purpose
```
# Activate virtual environment
source activate segmentation
# For segmentation task
python demo_segmentation.py
# For clustering task
python demo_clustering.py
```

Exploration - Graphic Element Classification and Text Extraction:
1. Download all files in demo/Exploration - Graphic Element Classification and Text Extraction
<https://git.unl.edu/unl_loc_summer_collab/codebase/tree/master/demo/Exploration%20-%20Graphic%20Element%20Classification%20and%20Text%20Extraction>
2. Install the required software and libraries
3. Download dataset from 
<https://git.unl.edu/unl_loc_summer_collab/labeled_data/tree/master/BeyondWord_orginal_resolution>
4. Copy the downloaded folder to the downloaded 'Exploration - Graphic Element Classification and Text Extraction' folder
5. Run the evaluation script
```
python eval.py
```

Exploration - Document Type Classification:
1. Download all files in demo/Exploration - Document Type Classification
<https://git.unl.edu/unl_loc_summer_collab/codebase/tree/master/demo/Exploration%20-%20Document%20Type%20Classification>
2. Install required softwares and libraries
3. Download dataset from 
<https://git.unl.edu/unl_loc_summer_collab/labeled_data/tree/master/suffrage_1002>
4. Copy the downlaoded folder to the downloaded 'Exploration - Document Type Classification' folder
5. Run the demonstration script
```
# Activate virtual environment
source activate Exploration - Document Type Classification
# Run demonstration
python demo_classification.py
```

Exploration - Digitization Type Differentiation:
1. Download all files in [demo/Exploration - Digitization Type Differentiation]
<https://git.unl.edu/unl_loc_summer_collab/codebase/tree/master/demo/Exploration%20-%20Digitization%20Type%20Differentiation>
2. Install the required software and libraries
3. Download dataset from 
<https://git.unl.edu/unl_loc_summer_collab/labeled_data/tree/master/micrpfilm_scanning>
4. Copy the downloaded folder to the downloaded 'Exploration - Digitization Type Differentiation' folder
5. Run the evaluation script
```
python eval.py
```

### Breaking down into end-to-end tests

Please read the README file inside each project folder for a description of each end-to-end test.

* Exploration - Document Segmentation: <https://git.unl.edu/unl_loc_summer_collab/codebase/tree/master/demo/Exploration%20-%20Document%20Segmentation>
* Exploration - Digitization Type Differentiation: <https://git.unl.edu/unl_loc_summer_collab/codebase/tree/master/Exploration%20-%20Graphic Element%20Classification%20and%20Text%20Extraction>
* Exploration - Document Type Classification: <https://git.unl.edu/unl_loc_summer_collab/codebase/tree/master/Exploration%20-%20Document%20Type%20Classification>
* Exploration - Document Image Quality Assessment: <https://git.unl.edu/unl_loc_summer_collab/codebase/tree/master/Exploration%20-%20Document%20Image%20Quality%20Assessment>
* Exploration - Digitization Type Differentiation: <https://git.unl.edu/unl_loc_summer_collab/codebase/tree/master/Exploration%20-%20Digitization%20Type%20Differentiation>

## Built With

* [Python](https://www.python.org/) - The programming language
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) - Enable GPU for model training
* [MXNet](https://mxnet.apache.org/) - Deep learning framework
* [TensorFLow](https://www.tensorflow.org/) - Deep learning framework
* [Matlab](https://www.mathworks.com/products/matlab.html) - Math, graphics, programming platform

## Contributing

| --- | Inputs | Technique | Output | Reports |
| --- | --- | --- | --- | --- |
| Exploration - Document Segmentation (segmentation)| ENP_500 (European historical newspaper)<br/> Beyond Words | U-Net | 5 class pixel-level segmented image| Progress report - Chulwoo Pack - 07312019.pdf<br/> Progress report - Chulwoo Pack - 08052019.pdf  |
| Exploration - Document Clustering (clustering)| ENP_500 (European historical newspaper) | t-SNE | Clustered manifold | Progress report - Chulwoo Pack - 09232019.pdf |
| Exploration - Graphic Element Classification and Text Extraction | ENP_500 (European historical newspaper)<br/> Beyond Words | U-NeXt | Predicted region segmentation | Progress report - Yi Liu - 07302019.pdf |
| Exploration - Document Type Classification | suffrage_1002 (LoC Suffrage campaign) | U-Net | Type of document image: handwritten, typed, and mixed | Progress report - Chulwoo Pack - 08132919<br/> Progress report - Chulwoo Pack - 08202019.pdf |
| Exploration - Document Image Quality Assessment | Civil War Campaign | DIQA | Four quality scores | Progress report - Yi Liu - 08122019.pdf<br/> Progress report - Yi Liu - 09052019.pdf |
| Exploration - Document Image Quality Assessment | difficulty_collection (LoC Manuscript/Mixed material) | U-Net, DIQA | visual difficulty correlation | Progress report - Chulwoo Pack - 10312019.pdf |
| Exploration - Digitization Type Differentiation | Civil War Campaign | ResNeXt | Classify micrpfilm or scanning | Progress report - Yi Liu - 09052019.pdf<br/> Progress report - Yi Liu - 10292019.pdf |


## Authors

* **Yi Liu** - *research associate and developer*
* **Chulwoo (Mike) Pack** - *research associate and developer*
* **Elizabeth Lorang** - *senior adviser*
* **Leen-Kiat Soh** - *senior adviser*
* **Ashlyn Stewart** - *research assistant*

## License

This project is licensed under the GPL License - see the [LICENSE](LICENSE) file for details

    