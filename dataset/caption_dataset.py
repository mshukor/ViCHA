import json
import os
import random
import torch 
from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption
import numpy as np
   
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
        print(len(self.ann))    
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
        print(len(self.ann))

        # self.ann = self.ann[:5000]

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            if isinstance(ann['caption'], list):
                for i, caption in enumerate(ann['caption']):
                    self.text.append(pre_caption(caption,self.max_words))
                    self.img2txt[img_id].append(txt_id)
                    self.txt2img[txt_id] = img_id
                    txt_id += 1
            else:
                caption = ann['caption']
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

        print(len(self.text))                  
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
            tmp =  json.load(open(f,'r'))
            self.ann += tmp
            print('size of', f, len(tmp))
        print(len(self.ann))
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
    max_seq_len = max([l.shape[1] for l in input_list])
    new_tensor = torch.zeros((len(input_list), max_seq_len)).long()

    output = [v + [0] * (max_seq_len - len(v)) for v in input_list]

    return torch.Tensor(output)




############################################################ KW img dataset

class pretrain_kw_img_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30, data_dir='/data/mshukor/data', 
        tokenizer=None, sep_token=True, randkw_p=None, num_kws=15):        
        self.ann = []
        for f in ann_file:
            tmp =  json.load(open(f,'r'))
            self.ann += tmp
            print('size of', f, len(tmp))
        self.transform = transform
        self.max_words = max_words
        for e in self.ann:
            e['image'] = os.path.join(data_dir, ('/').join(e['image'].split('/')[4:]))
            
        self.tokenizer = tokenizer
        self.sep_token = sep_token
        self.randkw_p = randkw_p
        print('self.randkw_p:', self.randkw_p)

        self.num_kws = num_kws
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        

        kwords = ann['kwords']


        if self.randkw_p is not None:
            num_kw = int(self.randkw_p * len(kwords))
            kws = random.choices(kwords, k=num_kw)
        else:
            kws = kwords

        text = dict()
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
      
         
        if self.sep_token:
            kw = ' [SEP] '.join(kws)
        else:
            kw = ' '.join(kws)
           


        image = Image.open(ann['image']).convert('RGB')   
        image = self.transform(image)
                
        return image, (caption, kw)
      

class re_kw_img_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, tokenizer=None, 
        sep_token=True, randkw_p=None, num_kws=15):        
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
        self.tokenizer = tokenizer
        self.sep_token = sep_token
        self.randkw_p = randkw_p
        
        self.num_kws = num_kws

    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        

        kwords = ann['kwords']


        if self.randkw_p is not None:
            num_kw = int(self.randkw_p * len(kwords))
            kws = random.choices(kwords, k=num_kw)
        else:
            kws = kwords
            
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = pre_caption(ann['caption'], self.max_words) 

        if self.sep_token:
            kw = ' [SEP] '.join(kws)
        else:
            kw = ' '.join(kws)

                
        return image, (caption, kw), self.img_ids[ann['image_id']]
    
    

class re_kw_img_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, tokenizer=None, 
        sep_token=True, num_kws=15):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        self.title = []
        self.kwords = []
        self.tokenizer = tokenizer

        self.sep_token = sep_token
        
        self.num_kws = num_kws

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            
            if isinstance(ann['caption'], list):
                for i, caption in enumerate(ann['caption']):
                    cap = pre_caption(caption,self.max_words)
                    

                    self.text.append([cap, cap])
                    self.img2txt[img_id].append(txt_id)
                    self.txt2img[txt_id] = img_id
                    txt_id += 1
            else:
                caption = ann['caption']
                cap = pre_caption(caption,self.max_words)


                self.text.append([cap, cap])
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  


        kws = self.ann[index]['kwords']


        if self.sep_token:
            kw = ' [SEP] '.join(kws)
        else:
            kw = ' '.join(kws)

        return (image, kw), index



