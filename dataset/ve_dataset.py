import json
import os
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption

import torch
import numpy as np 

class ve_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.labels = {'entailment':2,'neutral':1,'contradiction':0}
        print('size of dataset', len(self.ann))
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,'%s.jpg'%ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          

        sentence = pre_caption(ann['sentence'], self.max_words)

        return image, sentence, self.labels[ann['label']]



class ve_kw_img_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, sep_token=True, randkw_p=None):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.labels = {'entailment':2,'neutral':1,'contradiction':0}
        print('size of dataset', len(self.ann))

        self.sep_token = sep_token
        self.randkw_p = randkw_p

    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if self.randkw_p is not None:
            num_kw = int(self.randkw_p * len(ann['kwords']))
            kws = random.choices(ann['kwords'], k=num_kw)
        else:
            kws = ann['kwords']

        image_path = os.path.join(self.image_root,'%s.jpg'%ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          

        sentence = pre_caption(ann['sentence'], self.max_words)



        if self.sep_token:
            kw = ' [SEP] '.join(kws)
        else:
            kw = ' '.join(kws)

        return image, (sentence, kw), self.labels[ann['label']]

class ve_clip_token_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, data_dir='/data/mshukor/data', clip_embed_file=None):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.labels = {'entailment':2,'neutral':1,'contradiction':0}
        print('size of dataset', len(self.ann))

        self.clip_ann = {}
        tmp =  json.load(open(clip_embed_file,'r'))
        self.clip_ann.update(tmp)
            
        for k, v in self.clip_ann.items():
            v = os.path.join(data_dir, ('/').join(v.split('/')[4:]))

    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        

        image_path = os.path.join(self.image_root,'%s.jpg'%ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          

        sentence = pre_caption(ann['sentence'], self.max_words)



        clip_token = json.load(open(self.clip_ann[ann['image']] ,'r'))
        clip_token = torch.from_numpy(np.array(clip_token))

        return image, (sentence, clip_token), self.labels[ann['label']]