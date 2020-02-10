"""
Import modules
"""
import os
import requests
import pandas as pd
from enum import Enum 
from tqdm import tqdm
from nested_lookup import nested_lookup



"""
Utils
"""
def get_and_save_images(image_urls_list, image_filenames, folder_path):
    '''
    Takes as input a list of URLs for loc.gov item pages and 
    a path to a directory in which to save image files, e.g. "data". 
    '''
    for count, url in enumerate(image_urls_list):
        print("[{}/{}] downloading {}...".format(count,len(image_urls_list),url))
        try:
            # Set save file path
            identifier = image_filenames[count]
            filename = "{0}.jpg".format(identifier)
            filepath = os.path.join(folder_path, filename)
            # request the image and write to path
            full_url = "https:{0}".format(url)
            image_response = requests.get(full_url, stream=True)
            with open(filepath, 'wb') as fd:
                for chunk in image_response.iter_content(chunk_size=100000):
                    fd.write(chunk)
        except ConnectionError as e:
            print(e)


"""
Prepare save folder
"""
save_root = './suffrage_1002'
save_folder_path = os.path.join(save_root,'images')
if not os.path.exists(save_root):
    try:
        os.mkdir(save_root)
        os.mkdir(save_folder_path)
    except OSError:
        print("Creation of dir {} failed.".format(save_folder_path))
    else:
        print("Successfully created the dir {}.".format(save_folder_path))



"""
Read Excel file
"""
df = pd.read_excel('./image_url_with_gt.xlsx', sheetname=0)
urls = df['URL']
labels = df['Rating']



"""
Write labels.txt
"""
with open(os.path.join(save_root,'labels.txt'),'w') as fp:
    for idx,id in enumerate(file_names): 
        line = "images/{0}.jpg {1}\n".format(id,labels[idx])
        fp.write(line)



"""
Prepare image_urls and file_names
"""
image_urls = []
file_names = []
params = {"fo": "json"}

for url in tqdm(urls):
    # Generate image_url
    response = requests.get(url, params=params)
    try:
        data = response.json()
    except:
        print("Error occurs during parsing json: {}".format(url))
        continue
    image_url_sizes = nested_lookup('image_url',data)[0]
    if len(image_url_sizes)>IMAGE_SIZE.medium.value:
        image_url = image_url_sizes[IMAGE_SIZE.medium.value]
    else:
        image_url = image_url_sizes[0]
    image_urls.append(image_url)
    
    # Generate file_name
    if('tile' in image_url_sizes[0]):
        file_name = image_url_sizes[0].split('/')[5].replace(':','_')+('_'+url.split('=')[-1])
    else:
        file_name = image_url_sizes[1].split('/')[5].replace(':','_')+('_'+url.split('=')[-1])
    file_names.append(file_name)
    



"""
Download images
"""
get_and_save_images(image_urls, file_names, save_folder_path)
