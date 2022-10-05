import tarfile
import sys 
import os 
from tqdm import tqdm 
import json 

path = '/data/mshukor/data/cc12m/cc12m'
output_path = '/data/mshukor/data/our_albef_data/json_pretrain/cc12m.json'

data_ = []

for fold in tqdm(os.listdir(path)):
    dir_path = os.path.join(path, fold)
    if '.tar' in dir_path:
        tar = tarfile.open(dir_path)
        for member in tar.getmembers():
            if 'txt' in member.name:
                f=tar.extractfile(member)
                caption_path = os.path.join(f.name.replace('.tar', ''), member.name)
                caption = f.read().decode("utf-8")
                image_path = caption_path.replace('txt', 'jpg')
                d = {'image': image_path, 'caption': caption}
                data_.append(d)
        tar.close()
        
        
print('save:', output_path)
with open(output_path, 'w') as f:
    json.dump(data_, f)