import json
import os
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption


class nlvr_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root):        
        self.ann = []
        if not isinstance(ann_file, list):
            self.ann = json.load(open(ann_file,'r'))
        else:
            for f in ann_file:
                self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = 30
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image0_path = os.path.join(self.image_root,ann['images'][0])        
        image0 = Image.open(image0_path).convert('RGB')   
        image0 = self.transform(image0)   
        
        image1_path = os.path.join(self.image_root,ann['images'][1])              
        image1 = Image.open(image1_path).convert('RGB')     
        image1 = self.transform(image1)          

        sentence = pre_caption(ann['sentence'], self.max_words)
        
        if ann['label']=='True':
            label = 1
        else:
            label = 0

        return image0, image1, sentence, label



class nlvr_kw_img_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, sep_token=True, randkw_p=None):        
        self.ann = []
        if not isinstance(ann_file, list):
            self.ann = json.load(open(ann_file,'r'))
        else:
            for f in ann_file:
                self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = 30
        self.sep_token = sep_token
        self.randkw_p = randkw_p

    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if self.randkw_p is not None:
            num_kw = int(self.randkw_p * len(ann['kwords']))
            kws1 = random.choices(ann['kwords1'], k=num_kw)
            kws2 = random.choices(ann['kwords2'], k=num_kw)
        else:
            kws1 = ann['kwords1']
            kws2 = ann['kwords2']

        image0_path = os.path.join(self.image_root,ann['images'][0])        
        image0 = Image.open(image0_path).convert('RGB')   
        image0 = self.transform(image0)   
        
        image1_path = os.path.join(self.image_root,ann['images'][1])              
        image1 = Image.open(image1_path).convert('RGB')     
        image1 = self.transform(image1)          

        sentence = pre_caption(ann['sentence'], self.max_words)
        

        if self.sep_token:
            kw1 = ' [SEP] '.join(kws1)
            kw2 = ' [SEP] '.join(kws2)
        else:
            kw1 = ' '.join(kws1)
            kw2 = ' '.join(kws2)


        if ann['label']=='True':
            label = 1
        else:
            label = 0

        return image0, image1, (sentence, kw1, kw2), label