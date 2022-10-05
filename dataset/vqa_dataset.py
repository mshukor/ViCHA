import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from dataset.utils import pre_question
 
import torch 
import numpy as np 

class vqa_dataset(Dataset):
    def __init__(self, ann_file, transform, vqa_root, vg_root, eos='[SEP]', split="train", max_ques_words=30, answer_list=''):
        self.split = split        
        self.ann = []
        for f in ann_file:
            tmp = json.load(open(f,'r'))
            self.ann += tmp
            print(f, len(self.ann), len(tmp))
        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        self.max_ques_words = max_ques_words
        self.eos = eos
        
        if split=='test':
            self.max_ques_words = 50 # do not limit question length during test
            self.answer_list = json.load(open(answer_list,'r'))    
                
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if ann['dataset']=='vqa':
            image_path = os.path.join(self.vqa_root,ann['image'])    
        elif ann['dataset']=='vg':
            image_path = os.path.join(self.vg_root,ann['image'])  
            
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        if self.split == 'test':
            question = pre_question(ann['question'],self.max_ques_words)   
            question_id = ann['question_id']            
            return image, question, question_id


        elif self.split=='train':                       
            
            question = pre_question(ann['question'],self.max_ques_words)        
            
            if ann['dataset']=='vqa':
                
                answer_weight = {}
                for answer in ann['answer']:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1/len(ann['answer'])
                    else:
                        answer_weight[answer] = 1/len(ann['answer'])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())

            elif ann['dataset']=='vg':
                answers = [ann['answer']]
                weights = [0.5]  

            answers = [answer+self.eos for answer in answers]
                
            return image, question, answers, weights






class vqa_kw_img_dataset(Dataset):
    def __init__(self, ann_file, transform, vqa_root, vg_root, eos='[SEP]', split="train", max_ques_words=30, answer_list='', 
        tokenizer=None, sep_token=True, randkw_p=None, num_kws=15):
        self.split = split        
        self.ann = []
        for f in ann_file:
            tmp = json.load(open(f,'r'))
            self.ann += tmp
            print(f, len(self.ann), len(tmp))
            
        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        self.max_ques_words = max_ques_words
        self.eos = eos
        
        if split=='test':
            self.max_ques_words = 50 # do not limit question length during test
            self.answer_list = json.load(open(answer_list,'r'))    
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

        if ann['dataset']=='vqa':
            image_path = os.path.join(self.vqa_root,ann['image'])    
        elif ann['dataset']=='vg':
            image_path = os.path.join(self.vg_root,ann['image'])  
            
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        if self.split == 'test':
            question = pre_question(ann['question'],self.max_ques_words)   
            question_id = ann['question_id']   

            if self.sep_token:
                kw = ' [SEP] '.join(kws)
            else:
                kw = ' '.join(kws)

            return image, (question, kw), question_id


        elif self.split=='train':                       
            
            question = pre_question(ann['question'],self.max_ques_words)        
            
            if ann['dataset']=='vqa':
                
                answer_weight = {}
                for answer in ann['answer']:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1/len(ann['answer'])
                    else:
                        answer_weight[answer] = 1/len(ann['answer'])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())

            elif ann['dataset']=='vg':
                answers = [ann['answer']]
                weights = [0.5]  

            answers = [answer+self.eos for answer in answers]

            if self.sep_token:
                kw = ' [SEP] '.join(kws)
            else:
                kw = ' '.join(kws)
                
            return image, (question, kw), answers, weights

