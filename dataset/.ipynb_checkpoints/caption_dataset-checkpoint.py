import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = []
        
        if not isinstance(ann_file, list):
            self.ann = json.load(open(ann_file,'r'))
        else:
            for f in ann_file:
                self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']]
    
    

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
      
        

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30, data_dir='/data/mshukor/data'):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        for e in self.ann:
            e['image'] = os.path.join(data_dir, ('/').join(e['image'].split('/')[4:]))
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
      
        image = Image.open(ann['image']).convert('RGB')   
        image = self.transform(image)
                
        return image, caption
            

    
def list2Tensors(input_list):
    max_seq_len = max(map(len, input_list))
    output = [v + [0] * (max_seq_len - len(v)) for v in input_list]

    return torch.Tensor(output)

class pretrain_recipe_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30, data_dir='/data/mshukor/data', tokenizer=None):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        for e in self.ann:
            e['image'] = os.path.join(data_dir, ('/').join(e['image'].split('/')[4:]))
            
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        text = dict()
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
      
        
        caption = self.tokenizer(caption, padding='longest', truncation=True, max_length=25, return_tensors="pt")
        title = self.tokenizer(ann['title'], padding='longest', truncation=True, max_length=15, return_tensors="pt")
        kwords = [self.tokenizer(t, padding='longest', truncation=True, max_length=8, return_tensors="pt") for t in ann['kwords']]

        kwords = torch.cat(kwords, dim=0)
        
        image = Image.open(ann['image']).convert('RGB')   
        image = self.transform(image)
                
        return image, (caption, title, kwords)
  
