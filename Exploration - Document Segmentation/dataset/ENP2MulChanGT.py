"""
Load modules
"""
import os, sys
import numpy as np
import ntpath
from glob import glob
from PIL import Image, ImageDraw
from xml.dom import minidom
from shapely import geometry
from tqdm import tqdm


"""
Params
"""
# Note: Please aware that there are two version of PAGE XML, say old and new version. 
# Uncomment one of either code below accordingly.
XML_VER = "old"
#XML_VER = "new"
TRAIN_VAL_RATIO = 0.8



"""
Construct and create save directory structure
"""
ori_ENP_path = 'PATH/TO/ENP_images'
save_root = './ENP_500'
save_train_path = os.path.join(save_root,'train')
save_val_path   = os.path.join(save_root,'val')
save_train_images_path = os.path.join(save_train_path,'images')
save_train_labels_path = os.path.join(save_train_path,'labels')
save_val_images_path   = os.path.join(save_val_path,'images')
save_val_labels_path   = os.path.join(save_val_path,'labels')
if not os.path.exists(save_root):
    try:
        os.mkdir(save_root)
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



# Retrieve filenames for images and labels
image_filenames = sorted(glob(ori_ENP_path+"/*[!Gatos].tif"))
if(XML_VER=="old"):
	label_filenames = sorted(glob(ori_ENP_path+"/pc-*.xml"))
else:
	label_filenames = sorted(glob(ori_ENP_path+"/*pc.page.xml"))

# Retrieve image ids
image_ids = [ntpath.basename(image_filename).split('.')[0] for image_filename in image_filenames]

assert len(image_filenames)==len(image_ids), "Please check whether filename meets the following format: [0-9]*.tif"
assert len(image_filenames)==len(label_filenames), "The number of image and label files is not match:\n\timages: {}\n\tlabels: {}".format(len(image_filenames),len(label_filenames))

print("Total {} files are found".format(len(image_filenames)))



"""
Convert image format from tif to jpg and save under ./ENP_500/train/images 
and ./ENP_500/val/images accordingly.
"""
for idx in tqdm(range(len(image_filenames))):
    # Read as grayscale
    img = Image.open(os.path.join(ori_ENP_path,image_ids[idx]+".tif")).convert('L')
    # Convert to 3-channel
    img = np.stack((img,)*3, axis=-1)
    img = Image.fromarray(img)
    # Save under train/images
    if(idx<int(TRAIN_VAL_RATIO*len(image_filenames))):
        img.save(os.path.join(save_train_images_path,image_ids[idx]+".jpg"), "JPEG")
    # Save under val/images
    else:
        img.save(os.path.join(save_val_images_path,image_ids[idx]+".jpg"), "JPEG")
print("Images are saved.")


"""
Generate mask (ground-truth) and save under ./ENP_500/train/labels 
and ./ENP_500/val/labels accordingly.
"""
for idx in tqdm(range(len(image_filenames))):
    data = {}
    num_text = 0
    num_image = 0
    num_break = 0
    num_table = 0

    xmldoc = minidom.parse(label_filenames[idx])

    image_width  = int(xmldoc.getElementsByTagName('Page')[0].attributes['imageWidth'].value)
    image_height = int(xmldoc.getElementsByTagName('Page')[0].attributes['imageHeight'].value)

    textRegions = xmldoc.getElementsByTagName('TextRegion')
    imageRegions = xmldoc.getElementsByTagName('ImageRegion')
    lineDrawingRegions = xmldoc.getElementsByTagName('LineDrawingRegion')
    graphicRegions = xmldoc.getElementsByTagName('GraphicRegion')
    tableRegions = xmldoc.getElementsByTagName('TableRegion')
    chartRegions = xmldoc.getElementsByTagName('ChartRegion')
    separatorRegions = xmldoc.getElementsByTagName('SeparatorRegion')
    mathsRegions = xmldoc.getElementsByTagName('MathsRegion')
    noiseRegions = xmldoc.getElementsByTagName('NoiseRegion')
    frameRegions = xmldoc.getElementsByTagName('FrameRegion')
    unknownRegions = xmldoc.getElementsByTagName('UnknownRegion')
    totalRegions = textRegions + imageRegions + lineDrawingRegions + \
                   graphicRegions + tableRegions + chartRegions + \
                   separatorRegions + mathsRegions + noiseRegions + \
                   frameRegions + unknownRegions

    if False:
        print("{} \timageWidth".format(image_width))
        print("{} \timageHeight\n".format(image_height))

        print("{} \ttextRegion(s)".format(len(textRegions)))
        print("{} \timageRegion(s)".format(len(imageRegions)))
        print("{} \tlineDrawingRegion(s)".format(len(lineDrawingRegions)))
        print("{} \tgraphicRegion(s)".format(len(graphicRegions)))
        print("{} \ttableRegion(s)".format(len(tableRegions)))
        print("{} \tchartRegion(s)".format(len(chartRegions)))
        print("{} \tseparatorRegion(s)".format(len(separatorRegions)))
        print("{} \tmathsRegion(s)".format(len(mathsRegions)))
        print("{} \tnoiseRegion(s)".format(len(noiseRegions)))
        print("{} \tframeRegion(s)".format(len(frameRegions)))
        print("{} \tunknownRegion(s)\n".format(len(unknownRegions)))
        print("{} \ttotalRegions(s)\n\n+=============================+\n".format(len(totalRegions)))


    """GT mulChanMask"""
    canvas = Image.new('RGB', (image_width,image_height), 0)  # L: 8bit binary
                                                            # 0: filling with 0 
        
                                                                                        # RED
    mask_image = generateMulChanMask(canvas,textRegions,(255,0,0),5,XML_VER)            # -Text
    
                                                                                        # GREEN
    mask_image = generateMulChanMask(mask_image,imageRegions,(0,255,0),5,XML_VER)       # -image
    mask_image = generateMulChanMask(mask_image,graphicRegions,(0,255,0),5,XML_VER)     # -graphic
    mask_image = generateMulChanMask(mask_image,chartRegions,(0,255,0),5,XML_VER)       # -chart
                
                                                                                        # BLUE
    mask_image = generateMulChanMask(mask_image,separatorRegions,(0,0,255),5,XML_VER)   # separator
    
                                                                                        # YELLOW
    mask_image = generateMulChanMask(mask_image,tableRegions,(255,255,0),5,XML_VER)     # table
    
    mask_array = np.array(mask_image)
    
    # Save under train/labels
    if(idx<int(TRAIN_VAL_RATIO*len(image_filenames))):
    	mask_image.save(os.path.join(save_train_labels_path, image_ids[idx]+".png"), "PNG")
    # Save under val/labels
    else:
    	mask_image.save(os.path.join(save_val_labels_path, image_ids[idx]+".png"), "PNG")
    

print("Labels are saved.")
