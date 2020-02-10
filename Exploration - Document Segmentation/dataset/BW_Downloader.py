"""
Load modules
"""
import json
from collections import defaultdict
import os 
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm


"""
Utils
"""
def extract_values(obj,key):
    """Pull all values of specified key from nested JSON."""
    arr = []
    
    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v,arr,key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj,list):
            for item in obj:
                extract(item, arr, key)
        return arr
    
    results = extract(obj, arr, key)
    return results

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() 
                            if len(locs)>1)


"""
Construct and create save directory structure
"""
save_root = './Beyond_Words'
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

save_xml_root   = './bw_xmls'
if not os.path.exists(save_xml_root):
    try:
        os.mkdir(save_xml_root)
    except OSError:
        print("Creation of dir {} failed.".format(save_xml_root))
    else:
        print("Successfully created the dir {}.".format(save_xml_root))




"""
Read JSON
"""
with open('beyondwords_gt.json') as json_file:
    data = json.load(json_file)
image_locations = extract_values(data,'standard')
tot = len(image_locations)
flag = np.arange(tot).tolist()
# list of images to be downloaded
image_locations_dn = []
print("Total {} image entities in JSON.".format(tot))


"""
Main Process
"""
#Process a page with multiple images
count = 0
for dup in (list_duplicates(image_locations)):
    # to keep track of how many pages having multiple images
    count+=1
    
    # get page name and extension
    image_location,idxes = dup
    image_name = os.path.splitext(os.path.basename(image_location))[0]
    image_ext  = os.path.splitext(os.path.basename(image_location))[1]

    # set out xml filename
    namespace = image_location.split('/')
    out_filename = ''
    for tag in namespace[3:-1]:
        out_filename+=(tag+'_')
    out_filename+=image_name

    # create the file structure
    data_PcGts = ET.Element('PcGts')
    data_PcGts.set('xmlns','http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15')
    data_PcGts.set('xmlns:xsi','http://www.w3.org/2001/XMLSchema-instance')
    data_PcGts.set('xsi:schemaLocation','http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd')

    data_meta = ET.SubElement(data_PcGts, 'Metadata')

    data_page = ET.SubElement(data_PcGts,'Page')
    data_page.set("imageLocation",(image_location))
    data_page.set("imageFilename",(os.path.basename(image_location)))
    data_page.set("imageHeight",str(data['data'][idxes[0]]['height']))
    data_page.set("imageWidth",str(data['data'][idxes[0]]['width']))

    """Through this loop, collect coordinates of each image entity"""
    for idx in idxes:
        # remove ... from flag to keep track of single page with single image
        try:
            flag.remove(idx)
        except ValueError:
            pass
        try:
            x,y,width,height,label = int(data['data'][idx]['region']['x']), \
                                    int(data['data'][idx]['region']['y']), \
                                    int(data['data'][idx]['region']['width']), \
                                    int(data['data'][idx]['region']['height']), \
                                    extract_values(data['data'][idx],'category')[0]
            # inject coordinates
            data_imageRegion = ET.SubElement(data_page, 'ImageRegion')
            data_imageRegion.set('type',label)

            data_coords = ET.SubElement(data_imageRegion, 'Coords')        
            data_point = ET.SubElement(data_coords, 'Point')
            data_point.set('x',str(x))
            data_point.set('y',str(y))
            data_point = ET.SubElement(data_coords, 'Point')
            data_point.set('x',str(x+width))
            data_point.set('y',str(y))
            data_point = ET.SubElement(data_coords, 'Point')
            data_point.set('x',str(x+width))
            data_point.set('y',str(y+height))
            data_point = ET.SubElement(data_coords, 'Point')
            data_point.set('x',str(x))
            data_point.set('y',str(y+height))
        # exception handling for entity without "category"
        except IndexError:
            print("no category...{}".format(image_locations[idx]))
            continue
        # finalize file structure in xml format
        data_page_xml = ET.tostring(data_PcGts)
        #if(out_filename=="ndnp-jpeg-surrogates_uuml_indurain_ver01_data_sn85058396_print_1918072501_0275"):
            #print(out_filename)
        with open(save_xml_root+"/"+out_filename+".xml", "wb") as data_page_xml_file:
            data_page_xml_file.write(data_page_xml)

#Process a page with a single image
for idx in flag:
    # get page name and extension
    image_name = os.path.splitext(os.path.basename(image_locations[idx]))[0]
    image_ext  = os.path.splitext(os.path.basename(image_locations[idx]))[1]

    # set out xml filename
    namespace = image_locations[idx].split('/')
    out_filename = ''
    for tag in namespace[3:-1]:
        out_filename+=(tag+'_')
    out_filename+=image_name
    
    # create the file structure
    data_PcGts = ET.Element('PcGts')
    data_PcGts.set('xmlns','http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15')
    data_PcGts.set('xmlns:xsi','http://www.w3.org/2001/XMLSchema-instance')
    data_PcGts.set('xsi:schemaLocation','http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd')
    
    data_meta = ET.SubElement(data_PcGts, 'Metadata')
    
    data_page = ET.SubElement(data_PcGts,'Page')
    data_page.set("imageLocation",image_locations[idx])
    data_page.set("imageFilename",(os.path.basename(image_locations[idx])))
    data_page.set("imageHeight",str(data['data'][idx]['height']))
    data_page.set("imageWidth",str(data['data'][idx]['width']))
    
    try:
        x,y,width,height,label = int(data['data'][idx]['region']['x']), \
                                int(data['data'][idx]['region']['y']), \
                                int(data['data'][idx]['region']['width']), \
                                int(data['data'][idx]['region']['height']), \
                                extract_values(data['data'][idx],'category')[0]
   

        # inject coordinates
        data_imageRegion = ET.SubElement(data_page, 'ImageRegion')
        data_imageRegion.set('type',label)

        data_coords = ET.SubElement(data_imageRegion, 'Coords')        
        data_point = ET.SubElement(data_coords, 'Point')
        data_point.set('x',str(x))
        data_point.set('y',str(y))
        data_point = ET.SubElement(data_coords, 'Point')
        data_point.set('x',str(x+width))
        data_point.set('y',str(y))
        data_point = ET.SubElement(data_coords, 'Point')
        data_point.set('x',str(x+width))
        data_point.set('y',str(y+height))
        data_point = ET.SubElement(data_coords, 'Point')
        data_point.set('x',str(x))
        data_point.set('y',str(y+height))
    
    # exception handling for entity without "category"
    except IndexError:
        print("no category...{}".format(image_locations[idx]))
        continue
    
    # finalize file structure in xml format
    data_page_xml = ET.tostring(data_PcGts)
    with open(save_xml_root+"/"+out_filename+".xml", "wb") as data_page_xml_file:
        data_page_xml_file.write(data_page_xml)
    
    image_locations_dn.append(image_locations[idx])


tot_page = len(flag)+count
print("Total {} page(s) are processed.".format(tot_page))
"""Main done"""


"""
Download images from AWS
"""
save_image_root = "./bw_pages"
# Remove duplicated pages
image_locations_dn = list(dict.fromkeys(image_locations_dn))

for idx,image_location in tqdm(enumerate(image_locations_dn)):    
    # set output jpg filename
    image_name = os.path.splitext(os.path.basename(image_location))[0]
    namespace = image_location.split('/')
    out_filename = ''
    for tag in namespace[3:-1]:
        out_filename+=(tag+'_')
    out_filename+=image_name
    
    # Save under train/images
    if(idx<int(TRAIN_VAL_RATIO*len(image_locations_dn))):
        urllib.request.urlretrieve(image_location, os.path.join(save_train_images_path,out_filename+".jpg"))
    # Save under val/images
    else:
        urllib.request.urlretrieve(image_location, os.path.join(save_val_images_path,out_filename+".jpg"))


    
