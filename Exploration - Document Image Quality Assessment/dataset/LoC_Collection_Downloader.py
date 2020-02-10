"""
Load modules
"""
import csv
import json

import requests
import os
from urllib.parse import urlparse



"""
csv to JSON
"""
csvfile = open('assets.csv', 'r')
jsonfile = open('assets.json', 'w')
fieldnames = ('ID', 'item', 'published', 'title', 'slug', 'description', 'Path component of the URL', 'media type', 'sequence', 'year', 'resource url', 'download url', 'metadata', 'transcription status', 'difficulty')
reader = csv.DictReader(csvfile, fieldnames)
data = json.dumps( [ row for row in reader ] )  
jsonfile.write(data)  



"""
Read JSON
"""
with open('assets.json') as json_file:
    data = json.load(json_file)



"""
Params
"""
ROOT = './dataset' # Path to save images
MAX  = len(data)   # You can limit the number of images to download here



"""
Downloader
"""
for idx,item in enumerate(data):
    item_diff = int(item['difficulty']) 
    item_id = item['ID']
    item_dn_url = item['download url']
    
    save_path = os.path.join(ROOT,str(item_diff),(str(item_id)+'.jpg'))
    
    if not os.path.exists(os.path.join(ROOT,str(item_diff))):
        try:
            os.mkdir(os.path.join(ROOT,str(item_diff)))
        except OSError:
            print("Creation of dir {} failed.".format(os.path.join(ROOT,str(item_diff))))
        else:
            print("Successfully created the dir {}.".format(os.path.join(ROOT,str(item_diff))))
    
    print("[{}/{}] Downloading {}".format((idx+1),MAX,item_dn_url))
    # connect url
    try:
        image_response = requests.get(item_dn_url, stream=True)
        with open(save_path, 'wb') as fd:
            for chunk in image_response.iter_content(chunk_size=100000):
                fd.write(chunk)
    except ConnectionError as e:
        print(e)