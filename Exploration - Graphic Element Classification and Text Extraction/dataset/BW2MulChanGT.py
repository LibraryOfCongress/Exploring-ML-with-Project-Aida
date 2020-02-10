"""
Load modules
"""
import os, sys
from xml.dom import minidom
import numpy as np
from tqdm import tqdm

from glob import glob 

from PIL import Image, ImageDraw
from shapely import geometry


"""
Params
"""
TRAIN_VAL_RATIO = 0.8
XML_ROOT = "./bw_xmls"    # Path to where xml files are stored
GT_ROOT  = "./Beyond_Words"  # Path to save ground-truth
save_train_path = os.path.join(GT_ROOT,'train')
save_val_path   = os.path.join(GT_ROOT,'val')
save_train_labels_path = os.path.join(save_train_path,'labels')
save_val_labels_path   = os.path.join(save_val_path,'labels')


"""
Construct and create save directory structure
"""
if not os.path.exists(GT_ROOT):
    try:
        os.mkdir(GT_ROOT)
        os.mkdir(save_train_path)
        os.mkdir(save_val_path)
        os.mkdir(save_train_images_path)
        os.mkdir(save_train_labels_path)
        os.mkdir(save_val_images_path)
        os.mkdir(save_val_labels_path)
    except OSError:
        print("Creation of dir failed.")
    else:
        print("Successfully created the dirs.")



"""Utils"""
def generateMask(mask,regions):
    """Form a bit mask. Return an Image object.
    1. Convert a set of coordinates in xml to polygon
    2. Map each polygon into a bit mask

    Keyword arguments:
    mask    -- An Image object size of (imageWidth X imageHeight)
    regions -- A list of polygons
    """
    for region in regions:
        poly_points = []
        points = region.getElementsByTagName("Point")
        for point in points:
            x = float(point.attributes['x'].value)
            y = float(point.attributes['y'].value)
            poly_points.append(geometry.Point(x,y))
        polygon = geometry.Polygon([[p.x, p.y] for p in poly_points])
        ImageDraw.Draw(mask).polygon(list(polygon.exterior.coords), outline=1, fill=1)
    return mask


def generateMulChanMask(mulChanMask,regions,val):
    """Form a 3-channel ground truth mask. Return an Image object.
    
    Keyword arguments:
    mulChanGT  -- 
    regions    -- A list of polygons
    """
    for region in regions:
        poly_points = []
        points = region.getElementsByTagName("Point")
        for point in points:
            x = float(point.attributes['x'].value)
            y = float(point.attributes['y'].value)
            poly_points.append(geometry.Point(x,y))
        polygon = geometry.Polygon([[p.x, p.y] for p in poly_points])
        #ImageDraw.Draw(mulChanMask).rectangle(polygon.bounds,outline=0,fill=val)
        #x0,y0,x1,y1 = polygon.bounds
        #x0+=margin
        #y0+=margin
        #x1-=margin
        #y1-=margin
        #ImageDraw.Draw(mulChanMask).rectangle([x0,y0,x1,y1],outline=0,fill=val)
        
        ImageDraw.Draw(mulChanMask).polygon(list(polygon.exterior.coords),outline=val,fill=val)
    return mulChanMask



"""
Main Process
"""
file_list = glob(location_gt + "/*.xml")
print("Total {} files.".format(len(file_list)))

for file in tqdm(file_list):
    # get filename
    filename = os.path.splitext(os.path.basename(file))[0]
    # get xml
    xmldoc = minidom.parse(file)
    
    # parse xml info
    image_width  = int(xmldoc.getElementsByTagName('Page')[0].attributes['imageWidth'].value)
    image_height = int(xmldoc.getElementsByTagName('Page')[0].attributes['imageHeight'].value)

    imageRegions = xmldoc.getElementsByTagName('ImageRegion')

    editoCartoons = []
    comicCartoons = []
    illustrations = []
    photographs = []
    maps = []

    for imageRegion in imageRegions:
        imageType = imageRegion.attributes['type'].value
        if imageType == 'Editorial Cartoon':
            editoCartoons.append(imageRegion)
        elif imageType == 'Comics/Cartoon':
            comicCartoons.append(imageRegion)
        elif imageType == 'Illustration':
            illustrations.append(imageRegion)
        elif imageType == 'Photograph':
            photographs.append(imageRegion)
        elif imageType == 'Map':
            maps.append(imageRegion)
    
    """GT mulChanMask"""
    canvas = Image.new('RGB', (image_width,image_height), 0)  # L: 8bit binary
                                                            # 0: filling with 0 

    mask_image = generateMulChanMask(canvas,editoCartoons,(255,0,0))      # red
    mask_image = generateMulChanMask(mask_image,comicCartoons,(0,255,0))  # green
    mask_image = generateMulChanMask(mask_image,illustrations,(0,0,255))  # blue
    mask_image = generateMulChanMask(mask_image,photographs,(255,255,0))  # yellow
    mask_image = generateMulChanMask(mask_image,maps,(255,0,255))         # magenta

    mask_array = np.array(mask_image)

    """Save image"""
    # Save under train/labels
    if(idx<int(TRAIN_VAL_RATIO*len(image_filenames))):
        mask_image.save(os.path.join(save_train_labels_path,filename + ".png"), "PNG")
    # Save under val/labels
    else:
        mask_image.save(os.path.join(save_val_labels_path,filename + ".png"), "PNG")
    


    
