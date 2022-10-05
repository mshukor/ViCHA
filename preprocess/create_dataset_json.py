import json 
from tqdm import tqdm
import os 
from collections import defaultdict
from glob import glob
import random





def create_sbu_json(annot_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    data = json.load(open(annot_path,'r'))
    new_data = []
    
    for d in tqdm(data):
        
        new_dict = {'image': d[0].replace('images', 'images_train'), 'caption': d[1]}
        
        new_data.append(new_dict)
        
    out_path = os.path.join(output_dir, 'sbu.json')
    with open(out_path, 'w') as file:
        json.dump(new_data, file)
        
        
        
def to_dict_coco(path, iid2captions, iid2split, iid2id):
    name = path.split("/")[-1]
    captions = iid2captions[name]
    split = iid2split[name]
    id_ = iid2id[name]
    di = []
    for c in captions:
        di.append({'image': path, 'caption':c, 'image_id': id_})
    
    return split, di

def create_coco_json(data_dir, output_dir, split=['train', 'val'], output_file='coco.json'):
    with open(f"{data_dir}/karpathy/dataset_coco.json", "r") as fp:
        captions = json.load(fp)

    captions = captions["images"]

    iid2captions = defaultdict(list)
    iid2split = dict()
    iid2id = dict()

    for cap in tqdm(captions):
        filename = cap["filename"]
        iid2id[filename] = cap['cocoid']
        iid2split[filename] = cap["split"]
        for c in cap["sentences"]:
            iid2captions[filename].append(c["raw"])

    paths = list(glob(f"{data_dir}/train2014/*.jpg")) + list(glob(f"{data_dir}/val2014/*.jpg"))
    random.shuffle(paths)
    caption_paths = [path for path in tqdm(paths) if path.split("/")[-1] in iid2captions]

    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(
        len(paths), len(caption_paths), len(iid2captions),
    )

    new_data = []
    num=0
    other_splits = set()
    for path in tqdm(caption_paths):
        s, di = to_dict_coco(path, iid2captions, iid2split, iid2id)
        if s in split:
            num+=1
            for d in di:
                new_data.append(d)
        else:
            other_splits.add(s)
    print(split, num, 'images', ', other_splits:', other_splits)
    out_path = os.path.join(output_dir, output_file)
    with open(out_path, 'w') as file:
        json.dump(new_data, file)
        
        
def create_flicker_json(data_dir, output_dir, split=['train', 'val'], output_file='flicker.json'):
    with open(f"{data_dir}/karpathy/dataset_flickr30k.json", "r") as fp:
        captions = json.load(fp)

    captions = captions["images"]

    iid2captions = defaultdict(list)
    iid2split = dict()

    for cap in tqdm(captions):
        filename = cap["filename"]
        iid2split[filename] = cap["split"]
        for c in cap["sentences"]:
            iid2captions[filename].append(c["raw"])

    paths = list(glob(f"{data_dir}/flickr30k-images/*.jpg"))
    random.shuffle(paths)
    caption_paths = [path for path in tqdm(paths) if path.split("/")[-1] in iid2captions]

    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(
        len(paths), len(caption_paths), len(iid2captions),
    )

    new_data = []
    num=0
    other_splits = set()
    for path in tqdm(caption_paths):
        s, di = to_dict_coco(path, iid2captions, iid2split)
        if s in split:
            num+=1
            for d in di:
                new_data.append(d)
        else:
            other_splits.add(s)
    print(split, num, 'images', ', other_splits:', other_splits)
    out_path = os.path.join(output_dir, output_file)
    with open(out_path, 'w') as file:
        json.dump(new_data, file)
        

def to_dict_vg(path, iid2captions):
    name = path.split("/")[-1]
    iid = int(name[:-4])

    cdicts = iid2captions[iid]
    captions = [c["phrase"] for c in cdicts]

    di = []
    for c in captions:
        di.append({'image': path, 'caption':c})
        
    return di
        
def create_vg_json(data_dir, output_dir, split=['train', 'val']):
    with open(f"{data_dir}/annotations/region_descriptions.json", "r") as fp:
        captions = json.load(fp)

    iid2captions = defaultdict(list)
    for cap in tqdm(captions):
        cap = cap["regions"]
        for c in cap:
            iid2captions[c["image_id"]].append(c)

    paths = list(glob(f"{data_dir}/images/VG_100K/*.jpg")) + list(
        glob(f"{data_dir}/images/VG_100K_2/*.jpg")
    )
    random.shuffle(paths)
    caption_paths = [
        path for path in paths if int(path.split("/")[-1][:-4]) in iid2captions
    ]

    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(
        len(paths), len(caption_paths), len(iid2captions),
    )

    new_data = []
    for path in tqdm(caption_paths):
        di = to_dict_vg(path, iid2captions)
        for d in di:
            new_data.append(d)

    out_path = os.path.join(output_dir, 'vg.json')
    with open(out_path, 'w') as file:
        json.dump(new_data, file)
        
        
def create_vg_qa_json(data_dir='/data/mshukor/data/visual_genome', output_dir='/data/mshukor/data/our_albef_data', json_path='/data/mshukor/data/albef_data/data/vg_qa.json'):
    
    data_ = json.load(open(json_path,'r'))

    vg_image_paths = list(glob(f"{data_dir}/images/VG_100K/*.jpg")) + list(glob(f"{data_dir}/images/VG_100K_2/*.jpg"))
    img_id_dict = {}
    for p in vg_image_paths:
        img_id = p.split('/')[-1]
        img_id_dict[img_id] = ('/').join(p.split('/')[-3:])

    for d in tqdm(data_):
        if d['dataset'] == 'vg':
            img_name = d['image'].split('/')[-1]
            d['image'] = img_id_dict[img_name]
            
    out_path = os.path.join(output_dir, 'vg_qa.json')
    with open(out_path, 'w') as file:
        json.dump(data_, file)
        
