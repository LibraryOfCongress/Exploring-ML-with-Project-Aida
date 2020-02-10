# Voronoi-based Document Layout Complexity Analysis
Modified and written by Chulwoo Pack ([cpack@cse.unl.edu](mailto:cpack@cse.unl.edu))

This repo contains Efficient Voronoi-based Document Layout Complexity Analysis code (originally worked by K. Kise and modified by Faisal Shafait). 

## Requirements
The following python packages are required:
``` python
cv2   # python-opencv
json
numpy
```

You can install them either simply using [pip](https://pip.pypa.io/en/stable/) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) into your system.

## How to Use

1. Note that the entire complexity analysis is a two-step process.
- Execute *be* (c++) to run Voronoi-based page segmentation
     - in: <input_image_path>
     - out: spatial information of (1) connected component (*cc.txt*), (2) zones (*line.txt*), and (3) metadata in JSON format (*metadata.json*)
- Execute *voronoi_docplexity.py* (python) to run connected component & zone analysis.
     - in: <input_image_path>
     - out: *voronoi_docplexity_out.json*


2. For running this analysis on a batch, please follow the below:
```bash
# Set where your input image batch is located
~$ input_location="PATH/TO/BATCH/LOCATION"
~$ images=$(find ${input_location} -type f -name "*.jpg")  # You can the image type

# Activate virtual environment if you have one
source activate <virtenvname>

# Run analysis
for i in ${images}
do 
   # Print currently image being processed
   echo ${i};
   # Convert image to binary
   python binarization_morphological.py ${i};
   # Convert image type to tiff
   gm convert ./data/binary/bi_ ./data/binary/bi_tiff;
   # Run Voronoi Segmentation
   ./be ./data/binary/bi_.tiff
   # Run document layout complexity analysis
   python voronoi_docplexity.py ${i};
   # Run document layout complexity analysis
   cat ./data/metadata/metadata >> voronoi_docplexity_out.json
   ;
done
```

## References
[1] Voronoi page segmentation algorithm:
       K.Kise, A.Sato and M.Iwata,
       Segmentation of Page Images Using the Area Voronoi Diagram,
       Computer Vision and Image Understanding,
       Vol.70, No.3, pp.370-382, Academic Press, 1998.6.

[2] Color coding format and evaluation of Voronoi algorithm:
       F. Shafait, D. Keysers, T.M. Breuel,
       Performance Evaluation and Benchmarking of Six Page Segmentation Algorithms
       IEEE Transactions on Pattern Analysis and Machine Intelligence
       Vol.30, No.6, pp.941-954, June 2008

## Comments, Bug Reports
Please send them to [cpack@cse.unl.edu](mailto:cpack@cse.unl.edu)

## Note

This program contains the following public domain code as its parts:

(1) Voronoi page segmentation code by K. Kise.
  www.science.uva.nl/research/dlia/software/kise

   The note in Kise's Voronoi page segmentation code's README file is as follows:

   "The authors of this software are Akinori Sato, Motoi Iwata
   and Koichi Kise except for the parts listed in (2).
   Copyright (c) 1999 Akinori Sato, Motoi Iwata and Koichi Kise.
   Permission to use, copy, modify, and distribute this software for any
   purpose without fee is hereby granted, provided that this entire notice
   is included in all copies of any software which is or includes a copy
   or modification of this software and in all copies of the supporting
   documentation for such software.
   THIS SOFTWARE IS BEING PROVIDED "AS IS",
   WITHOUT ANY EXPRESS OR IMPLIED WARRANTY. "

(2) Steve Fortune's Voronoi program: sweep2
  
  http://netlib.bell-labs.com/netlib/voronoi/index.html

   
