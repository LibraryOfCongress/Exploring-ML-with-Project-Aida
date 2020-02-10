# Set where your input image is located
#/home/cpack/aida/burney-collection-images-and-metadata/Burney/B0001ORIWEEJO/WO2_B0001ORIWEEJO_1715_11_19/WO2_B0001ORIWEEJO_1715_11_19-0001.tif
input_location="/home/cpack/aida/yil/TenKDownloader/TenK-pages-1834-1922"
images=$(find ${input_location} -type f -name "*.jp2")

# Activate virtual environment
source ../../../Quality_Assessment/TenK_Assessment/ImageProcessing/bin/activate

for i in ${images}
do 
	# Convert image to binary
	python binarization_morphological.py ${i}

	# Convert jp2 to tiff
	gm convert ./data/binary/bi_ ./data/binary/bi_.tiff

	# Run Voronoi Segmentation
	./be ./data/binary/bi_.tiff

	# Run document layout complexity analysis
	python voronoi_docplexity.py 

	# Collect generated analysis result
	cat ./data/metadata/metadata >> voronoi_docplexity_out.txt
	;
done