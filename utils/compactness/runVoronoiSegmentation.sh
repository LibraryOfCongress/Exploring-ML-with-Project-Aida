# Set where your input image is located
input_location="/Users/Mike/git/Voronoi_SegNews/data/"
# Run zoning
#start=`date +%s`
for i in ${input_location}*; do ./be ${i} ./output/textline/$(basename ${i%.*}'.txt');done
#end=`date +%s`
#runtime=$((end-start))
#echo 'run-time: '$runtime

# Drawing zones
for i in ./output/textline/*; do drawing/dl -i ${input_location}$(basename ${i%.*}'.tiff') -l ${i} -o ./output/zone/$(basename ${i%.*}'.tiff');done
#end=`date +%s`
#runtime=$((end-start))
#echo 'run-time: '$runtime
