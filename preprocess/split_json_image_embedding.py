import os
import json 
import torch
import clip
from PIL import Image
import sng_parser
from tqdm import tqdm 
import codecs
import numpy as np


def split_json_image_embeddings(json_path, output_path=None, 
    data_dir='/data/mshukor/data/our_albef_data/clip_da/image_embeddings/',):
    data = json.load(open(json_path,'r'))
    image_to_feat = {}
    os.makedirs(data_dir, exist_ok=True)
    for i, (k, d) in tqdm(enumerate(data.items())):
        path = os.path.join(data_dir, k.replace('/data/mshukor/data/', '')).replace('.jpg', '').replace('.png', '')+'.json'
        image_to_feat[k] = path
        os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
        json.dump(d, codecs.open(path, 'w', encoding='utf-8'))
    with open(output_path, 'w') as f:
        json.dump(image_to_feat, f)
    return image_to_feat
        
######## COCO
# json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/coco_image_embeddings.json'
# data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/coco.json'

# tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)

######## SBU
# json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/sbu_image_embeddings.json'
# data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/sbu.json'

# tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)

######## VG
# json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/vg_image_embeddings.json'
# data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/vg.json'

# tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)

###### CC3M
# json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/cc3m_image_embeddings.json'
# data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/cc3m.json'

# tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)

# json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/cc3m_val_image_embeddings.json'
# data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/cc3m_val.json'

# tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)


######## Flickr
# json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/flickr30k_train_image_embeddings.json'
# data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/flickr30k_train.json'

# tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)


# json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/flickr30k_val_image_embeddings.json'
# data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/flickr30k_val.json'

# tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)


# json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/flickr30k_test_image_embeddings.json'
# data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/flickr30k_test.json'

# tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)


###### VQA

# json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/coco_image_embeddings.json'
# data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/vqa_train.json'

# tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)

# json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/vqa_test_image_embeddings.json'
# data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/vqa_test.json'

# tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)


# json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/vqa_val_image_embeddings.json'
# data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/vqa_val.json'

# tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)


# json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/vg_image_embeddings.json'
# data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/vg_qa.json'

# tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)


## COCO
# json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/coco_karp_image_embeddings.json'
# data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/coco_karp.json'

# tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)

# json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/coco_test_karp_image_embeddings.json'
# data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/coco_karp_test.json'

# tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)

# json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/coco_val_karp_image_embeddings.json'
# data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/coco_karp_val.json'

# tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)

# json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/coco_karp_image_embeddings.json'
# data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/coco_karp_train.json'

# tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)


##### VE

# json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/ve_train_image_embeddings.json'
# data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/snli/data/images'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/ve_train.json'

# tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)


# json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/ve_test_image_embeddings.json'
# data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/snli/data/images'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/ve_test.json'

# tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)


# json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/ve_dev_image_embeddings.json'
# data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/snli/data/images'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/ve_dev.json'

# tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)


##### NLVR


json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/nlvr_train_image_embeddings.json'
data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/nlvr'
output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/nlvr_train.json'

tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)


json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/nlvr_test_image_embeddings.json'
data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/nlvr'
output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/nlvr_test.json'

tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)


json_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/nlvr_dev_image_embeddings.json'
data_dir = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/nlvr'
output_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/config/nlvr_dev.json'

tmp = split_json_image_embeddings(json_path, data_dir=data_dir, output_path=output_path)