import sys
import os
import re
from datetime import datetime

import cv2
import math
import numpy

import sauvola   # For Sauvola Binarization

from PIL import Image # For save binary image

import ntpath


def main():
    ##############
    # Get images #
    ##############
    imagePath = sys.argv[1]
    outputPath = os.path.abspath("./data/binary/")
    
    #################
    # Start process #
    #################
    greyscaleImage = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    t_sauvola = sauvola.threshold_sauvola(greyscaleImage, window_size=15, k=0.2)
    binaryImage = greyscaleImage > t_sauvola
    binaryImage = (binaryImage*255).astype(numpy.uint8)
    binaryImage = cv2.bitwise_not(binaryImage)

    # Morphological Transformation
    bw = binaryImage
    kernel = numpy.ones((3,1), numpy.uint8)  # note this is a horizontal kernel
    d_im = cv2.dilate(bw, kernel, iterations=1)
    e_im = cv2.erode(d_im, kernel, iterations=1) 
    binaryImage = e_im

    #save
    mode='1'
    binaryImage_ = Image.fromarray(cv2.bitwise_not(binaryImage))
    binaryImage_ = binaryImage_.convert('1')
    output_filename = "bi_"
    #cv2.imwrite(os.path.join(os.path.abspath("./output"), output_filename + ntpath.basename(outputPath)), binaryImage_)
    
    #binaryImage_.save(os.path.join(outputPath, output_filename + ntpath.basename(imagePath)), format='TIFF', dpi=(300.,300.), compression='tiff_lzw')
    binaryImage_.save(os.path.join(outputPath, output_filename), format='TIFF', dpi=(300.,300.), compression='tiff_lzw')

main()