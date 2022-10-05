import os
import json 
import torch
import clip
from PIL import Image
import sng_parser
from tqdm import tqdm 
import codecs
import numpy as np


def split_json_image_embeddings(json_path, output_path=None, data_dir='/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'):
    data = json.load(open(json_path,'r'))
    image_to_feat = {}
    os.makedirs(data_dir, exist_ok=True)
    for i, (k, d) in tqdm(enumerate(data.items())):
        path = os.path.join(data_dir, k.replace('/data/mshukor/data/', '')).replace('jpg', 'json')
        image_to_feat[k] = path
        os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
        json.dump(d, codecs.open(path, 'w', encoding='utf-8'))
    with open(output_path, 'w') as f:
        json.dump(image_to_feat, f)
    return image_to_feat
        
    
# json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/coco_image_embeddings.json'
# data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/coco.json'

# tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)


# json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/sbu_image_embeddings.json'
# data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/sbu.json'

# tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)


# json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/vg_image_embeddings.json'
# data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/vg.json'

# tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)


json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/flickr30k_train_image_embeddings.json'
data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'
output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/flickr30k_train.json'

tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)


json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/flickr30k_val_image_embeddings.json'
data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'
output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/flickr30k_val.json'

tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)


json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/flickr30k_test_image_embeddings.json'
data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'
output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/flickr30k_test.json'

tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)