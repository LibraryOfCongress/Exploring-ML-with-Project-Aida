"""
Document Image Quality Assessment (DIQA).
This code aims to measure the following four different quality metrics:
(1) skewness
(2) contrast
(3) range-effect
(4) bleed-through (or noise)
"""

# Modules
import sys
import os
import cv2
import numpy as np
from scipy.ndimage import interpolation
from scipy.ndimage import gaussian_filter1d
import sauvola   # For Sauvola Binarization


"""estimate_skew
This function aims to measure the skewness in document image range from (-maxskew,maxskew)
"""
def _estimate_skew_angle(image,angles):
    estimates = []
    for a in angles:
        v = np.mean(interpolation.rotate(image,a,order=0,mode='constant'),axis=1)
        v = np.var(v)
        estimates.append((v,a))
    _,a = max(estimates)
    return a

def estimate_skew(imagepath,bignore=0.1,maxskew=2,skewsteps=8):
    img = cv2.imread(imagepath,cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image is empty...")
        raise ValueError
    d0,d1 = img.shape
    o0,o1 = int(bignore*d0),int(bignore*d1) # border ignore
    img = np.amax(img)-img
    img -= np.amin(img)
    est = img[o0:d0-o0,o1:d1-o1]
    ma = maxskew
    ms = int(2*maxskew*skewsteps)
    # print(linspace(-ma,ma,ms+1))
    angle = np.around(_estimate_skew_angle(est,np.linspace(-ma,ma,ms+1)), decimals=3)
    return angle


"""estimate_contrast
This function aims to measure the contrast of document using adaptive gradient of contrast.
"""
def estimate_contrast(imagepath):
    # Read image
    img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image is empty...")
        raise ValueError
    # Binarize image
    #ret, img = cv2.threshold(cv2.medianBlur(img,5),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    greyscaleImage = img
    t_sauvola = sauvola.threshold_sauvola(greyscaleImage, window_size=15, k=0.2)
    binaryImage = greyscaleImage > t_sauvola
    binaryImage = (binaryImage*255).astype(np.uint8)
    binaryImage = cv2.bitwise_not(binaryImage)
    img = binaryImage
    
    # Get min and max value within neighboring pixels
    d_4 = np.roll(img,1,axis=1)
    d_6 = np.roll(img,-1,axis=1)
    d_8 = np.roll(img,1,axis=0)
    d_2 = np.roll(img,-1,axis=0)

    d_1 = np.roll(d_4,-1,axis=0)
    d_3 = np.roll(d_6,-1,axis=0)
    d_7 = np.roll(d_4,1,axis=0)
    d_9 = np.roll(d_6,1,axis=0)

    neighbor_stack = np.stack((d_1,d_2,d_3,d_4,img,d_6,d_7,d_8))

    local_max = np.max(neighbor_stack,axis=0)
    local_min = np.min(neighbor_stack,axis=0)
    # Get contrast
    contrast = (local_max-local_min)[1:-1,1:-1] # ignore 1 pixel of border
    # Get gradient of contrast
    gradient = cv2.Laplacian(contrast,cv2.CV_64F)
    # Get CG-score
    cg_score = np.around(np.mean(abs(gradient)), decimals=3)
    
    return cg_score



"""estimate_rangeeffect
This function aims to measure the range-effect in document using statistical analysis on the range of pixel-value in both horizontal and vertical direction.
"""
def estimate_rangeeffect(imagepath):
    EPS = 1e-2
    img = cv2.imread(imagepath,cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image \"{}\" is empty...".format(imagepath))
        raise ValueError
    h,w = np.shape(img)
    
    margin_h = int(h*0.15)
    margin_w = int(w*0.15)

    #max_vertical = np.clip(np.max(img[margin_w:-margin_w,margin_h:-margin_h],axis=0),0,255)
    #min_vertical = np.clip(np.min(img[margin_w:-margin_w,margin_h:-margin_h],axis=0),0,50)
    max_vertical = []
    min_vertical = []
    for col in range(margin_w,w-margin_w):
        strip = img[margin_h:-margin_h,col]
        max_vertical.append(int(np.max(strip)))
        min_vertical.append(np.min(strip)/(256+EPS-len(np.unique(strip))))


    diff_maxmin_ver = np.array(max_vertical)-np.array(min_vertical)
    std_ver = np.std(diff_maxmin_ver)

    #max_horizontal = np.clip(np.max(img[margin_w:-margin_w,margin_h:-margin_h],axis=1),0,255)
    #min_horizontal = np.clip(np.min(img[margin_w:-margin_w,margin_h:-margin_h],axis=1),0,50)
    max_horizontal = []
    min_horizontal = []
    for col in range(margin_w,w-margin_w):
        strip = img[margin_h:-margin_h,col]
        max_horizontal.append(int(np.max(strip)))
        min_horizontal.append(np.min(strip)/(256+EPS-len(np.unique(strip))))

    diff_maxmin_hor = np.array(max_horizontal)-np.array(min_horizontal)
    std_hor = np.std(diff_maxmin_hor)

    max_std = np.around(max(std_ver,std_hor),decimals=3)

    return max_std



"""estimate_bleedthrough
This function aims to measure degree of bleed-through in document using histogram analysis. 
"""
def estimate_bleedthrough(imagepath):
    WINDOW_RATIO = 0.9
    
    img = cv2.imread(imagepath,cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image \"{}\" is empty...".format(imagepath))
        raise ValueError
    h,w = np.shape(img)
    
    # Ignore boundary
    img_nb = img[int(h*0.1):-int(h*0.1),int(w*0.1):-int(w*0.1)]
        
    # Get 4 sub-images
    h_nb,w_nb = np.shape(img_nb)
    img_nb_subs = []
    img_nb_subs_2 = []
    for col in range(2):
        for row in range(2):        
            h_center = h_nb//4*(2*row+1)
            w_center = w_nb//4*(2*col+1)
            img_nb_subs.append(img_nb[h_center-int(w_nb/4*(WINDOW_RATIO/2)):h_center+int(w_nb/4*(WINDOW_RATIO/2)),w_center-int(w_nb/4*(WINDOW_RATIO/2)):w_center+int(w_nb/4*(WINDOW_RATIO/2))])    

    # Get First-threshold and binary image
    ths_1 = []
    bin_1 = []
    for img_nb_sub in img_nb_subs:
        ths,img_bin = cv2.threshold(cv2.medianBlur(img_nb_sub,5),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ths_1.append(ths)
        bin_1.append(img_bin)
            
    # Get Second-threshold and binary image
    ths_2 = []
    bin_2 = []
    diff_maps = []
    diff_maps_refined = []
    diff_means = []
    diff_props = []
    
    for i in range(4): 
        ignore_dark_pixels = []
        for col in range(np.shape(img_nb_subs[i])[1]):
            for row in range(np.shape(img_nb_subs[i])[0]):
                '''
                condition(1), img_nb_subs[i][row,col]>=ths_1[i]:
                    This condition filters out pixels with a value lower than the first-threshold. Thus, second-threshold for capturing bleed-through region can be generated on the brighter value.
                condition (2), img_nb_subs[i][row,col]!=255 and img_nb_subs[i][row,col]!=0:
                    This condition filters out pixels with a value of extremely high or low value, 255 and 0, respectively. Some images are showing anormal histogram having too many 255 and 0 values compared to other values. Since this unbalanced pixels interfere finding second-thresold, they are filtered out.
                '''
                if img_nb_subs[i][row,col]>=ths_1[i]:# and img_nb_subs[i][row,col]!=255 and img_nb_subs[i][row,col]!=0:
                    ignore_dark_pixels.append(img_nb_subs[i][row,col])
                    
        # Erode first binarized image
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(bin_1[i],kernel,iterations = 2)
        bin_1[i] = erosion

        #Get histogram from [1-threshold:end]
        hist = np.histogram(np.array(ignore_dark_pixels),255,[0,255])
        #plt.gcf().clear()

        # Plot smoothen histogram
        hist_gaussian = gaussian_filter1d(hist[0], 5)

        # Get second threshold
        peak = np.argmax(hist_gaussian)
        #ths, img_bin_2 = cv2.threshold(cv2.medianBlur(np.array(ignore_dark_pixels),5),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #ths = int((2*peak+1*ths_1[i])/3) #(1/3)
        max_pixel = np.max(ignore_dark_pixels)
        ths = peak-(max_pixel-peak)

        # Generate second binarization image
        img_bin_2 = np.copy(img_nb_subs[i])
        img_bin_2[img_bin_2<ths] = 0
        img_bin_2[img_bin_2>=ths] = 255

        #ths, img_bin_2 = cv2.threshold(cv2.medianBlur(img_nb_subs[i],5),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ths_2.append(ths)
        bin_2.append(img_bin_2)
        
        #Generate diff_map calculating differences between first- and second-binarized image
        diff_map = bin_1[i]-bin_2[i]
        diff_map[diff_map<255] = 0
        diff_maps.append(diff_map)

        #Calibrate differences
        diff_means.append(np.mean(diff_map))
        diff_props.append(len(diff_map[diff_map==255])/float(np.size(diff_map)))

    #Log results
    return np.around(np.mean(diff_means),decimals=3)
