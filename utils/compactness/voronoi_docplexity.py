import cv2
import os, sys
import numpy as np
import json

def exception_handler(exception_type, exception, traceback):
    # All your trace are belong to us!
    # your format
    #print("%s: %s".format(exception_type.__name__, exception))
    pass

sys.excepthook = exception_handler

def main(argv):
    input_image_path = str(argv[0])


    ##########
    # PARAMS #
    ##########
    CC_THRESHOLD = 5
    linesegments_location  = "./data/linesegments/"
    ccs_location           = "./data/ccs/"
    meta_location          = "./data/metadata/"

    in_linesegments_name = "line"
    in_cc_name           = "cc"
    in_meta_name         = "metadata"

    '''Read metadata'''

    # Read and process JSON
    file_metadata = open(os.path.join(meta_location,in_meta_name),"r")
    str_metadata = file_metadata.read()   
    try:
        json_metadata = json.loads(str_metadata)
    except json.decoder.JSONDecodeError:
        print("Cannot load a metadata JSON file...")
        #sys.exit()
        

    h = int(json_metadata['height'])
    w = int(json_metadata['width'])

    '''Read line segment info'''
    file_linesegments = open(os.path.join(linesegments_location,in_linesegments_name),"r")

    '''Build voronoi-ridge mask'''
    img_voronoi_mask = 255*np.ones([h,w])
    for line in file_linesegments: 
        # Parse coordinates
        cs,ce,rs,re = (int(x) for x in line.split())
        # Draw line
        cv2.line(img_voronoi_mask,(cs,rs),(ce,re),(0,0,0),5)

    '''Voronoi mask analysis'''
    count,label,stat,centroids=cv2.connectedComponentsWithStats((img_voronoi_mask).astype(np.uint8))


    '''Read cc info'''
    file_ccs = open(os.path.join(ccs_location,in_cc_name),"r")


    '''Build cc mask and caculate area of each zone of (img_voronoi_mask-img_cc_mask)'''
    img_cc_mask = 255*np.ones([h,w])
    area_minus_cc_mask = np.copy(stat[:,4])
    for line in file_ccs: 
        # Parse coordinates
        c,r = (int(x) for x in line.split())
        # Build cc mask
        img_cc_mask[r,c]=0
        # Calculate area
        area_minus_cc_mask[label[r,c]]-=1
        

    '''Combine voronoi mask and cc mask'''
    img_comb = cv2.bitwise_and(img_voronoi_mask,img_cc_mask)


    '''Find regions containing single cc'''
    """
    num_of_noise_zone = 0
    count_comb,label_comb,stat_comb,centroids_comb=cv2.connectedComponentsWithStats((img_comb).astype(np.uint8))
    if count_comb!=count: # or use assert?
        print("Unexpected error has occurred.")
    else:
        for i in range(count):
            area_diff = stat[i][4]-stat_comb[i][4]
            if(area_diff<=CC_THRESHOLD):
                num_of_noise_zone+=1
    """
    num_of_noise_zone = -1
    num_of_noise_zone = len(np.argwhere(stat[:,4]-area_minus_cc_mask<=CC_THRESHOLD))

    '''Analyze detected zones'''
    zone_area_dist = stat[:,4]

    zone_min  = np.min(zone_area_dist)
    zone_max  = np.max(zone_area_dist)
    zone_mean = int(np.mean(zone_area_dist))
    zone_std  = int(np.std(zone_area_dist))
    zone_median = int(np.median(zone_area_dist))

    zone_q1  = np.percentile(zone_area_dist, 25)
    zone_q3  = np.percentile(zone_area_dist, 75)
    zone_iqr = zone_q3-zone_q1

    '''Build JSON'''
    json_metadata['fileName']       = input_image_path
    json_metadata['numOfZone']      = str(count)
    json_metadata['numOfNoiseZone'] = str(num_of_noise_zone)
    ##
    json_metadata['zoneMin'] = str(zone_min)
    json_metadata['zoneMax'] = str(zone_max)
    json_metadata['zoneMean'] = str(zone_mean)
    json_metadata['zoneStd'] = str(zone_std)
    json_metadata['zoneMedian'] = str(zone_median)
    ##
    json_metadata['zoneQ1'] = str(zone_q1)
    json_metadata['zoneQ3'] = str(zone_q3)
    json_metadata['zoneIQR'] = str(zone_iqr)

    '''Overwrite JSON'''
    with open(os.path.join(meta_location,in_meta_name),"w") as json_outfile:
        json.dump(json_metadata,json_outfile)

if __name__ == "__main__":
    main(sys.argv[1:])

