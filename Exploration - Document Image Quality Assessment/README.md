# Introduction

"Exploration - Document Image Quality Assessment" and "Exploration - Advanced Document Image Quality Assessment" aims to analyze the image quality of the civil war collection By the People.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

For "Exploration - Document Image Quality Assessment" (first and second iteration), the required software and libraries are:
* Python 3.7
* scipy 1.3.1
* opencv-python 4.1
* skimage 0.15

For "Exploration - Advanced Document Image Quality Assessment" (third iteration), the required software and libraries are:
* Python 3.7
* opencv-python 4.1
* scikit-learn >= 0.20.3
* scikit-image >= 0.15.0
* matplotlib >= 1.4.3
* pandas >= 0.24.2
* seaborn 0.9.0

### Installing

A step by step series of examples that tell you how to install the required software and libraries.

For the first and second iteration:
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

For the third iteration:
1. Download Python 3.7 from <https://www.python.org/downloads/>
2. Install Anaconda or Miniconda ([installation procedure](https://conda.io/docs/user-guide/install/index.html#))
3. Open Terminal (for MacOS), Command-Line (for Windows)
4. Create a virtual environment and activate it
```
conda create -n diqa python=3.6
source activate diqa
```
5. Install packages
```
python setup.py install
```

## Data Acquisition
For the first and second iteration:
The dataset was downloaded from the Library of Congress. To download the collectio, please refer to [the Civil War collection on By The People](https://crowd.loc.gov/topics/civil-war/). 

For the third iteration, we will download a set of document images from the Library of Congress collections using our custom downloader script, which is based on [loc.gov JSON API](https://libraryofcongress.github.io/data-exploration/).
```
source activate
python LoC_Collection_Downloader.py
```
Our custom script downloads and saves images under the user-defined local directory, which will be created by the following structure:
```
.user-defined-directory-name
├── 3                # difficuly score
│   ├── 1440.jpg     # image_id.jpg
│   ├── ...      
│   └── 2340.jpg 
├── 5                # difficuly score
├── ...
└── 130              # difficuly score
```

## Running the DIQA (First and second iteration)

1. Configurate the training
    1.1 set the path to save the training log
1. Download dataset from 
<https://git.unl.edu/unl_loc_summer_collab/labeled_data/tree/master/civil_war>
3. Copy the downloaded folder to the downloaded 'project4' folder
4. Run the DIQA script
```
python run_diqa.py
```

### Formats of the training log

DIQA generates 'diqa_rslt.csv' to save the log.
```
diqa_rslt.csv
```
Each line of the log consists of four parts.
They are:
1. the file name and the saving path;
2. the skewness evaluation score;
3. the contrast evaluation score;
4. the range-effect evaluation score;
5. the bleed-through evaluation score;

## Running the difficulty analysis (Third iteration)
1. Prepare the dataset and locate it under the 'Exploration - Document Image Quality Assessment' 
2. Activate the virtual environment and run the analysis script
```
source activate diqa
python run_difficulty.py
```

## Built With

* [Python](https://www.python.org/) - The programming language

## Contributing

Evaluate the overall condition of the Civil War collection for further processes.

## Authors

* **Yi Liu** - University of Nebraska-Lincoln - *email* - yil@cse.unl.edu
* **Chulwoo Pack** - University of Nebraska-Lincoln - *email* - cpack@cse.unl.edu
