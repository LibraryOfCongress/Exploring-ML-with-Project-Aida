# Set where your input image is located
input_location="/Users/Mike/GoogleDrive/Research/2017Aida/Page_Segmentation/Column_Segmentation/Coding/Binarization/outputs/converted/GA/"
# Run zoning
start=`date +%s`
for i in ${input_location}*; do ./be ${i} ./output/textline/$(basename ${i%.*}'.txt');done
end=`date +%s`
runtime=$((end-start))
echo 'run-time: '$runtime

# Drawing zones
for i in ./output/textline/*; do drawing/dl -i ${input_location}$(basename ${i%.*}'.tiff') -l ${i} -o ./output/zone/$(basename ${i%.*}'.tiff');done
end=`date +%s`
runtime=$((end-start))
echo 'run-time: '$runtime

# 25/11/18 16:44:11 run zoning
# 

# DPI: 300
# BC: 25
# CA: 64
# GA: 19
# SV: 11

# DPI: 600
#
# CA: 
