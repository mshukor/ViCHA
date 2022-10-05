import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.caption_dataset import re_train_dataset, re_eval_dataset, pretrain_dataset, \
pretrain_kw_img_dataset, re_kw_img_train_dataset, re_kw_img_eval_dataset

from dataset.nlvr_dataset import nlvr_dataset, nlvr_kw_img_dataset

from dataset.ve_dataset import ve_dataset, ve_kw_img_dataset

from dataset.vqa_dataset import vqa_dataset, vqa_kw_img_dataset

from dataset.grounding_dataset import grounding_dataset, grounding_kw_img_dataset

from dataset.randaugment import RandomAugment
 
def create_kw_img_dataset(dataset, config, data_dir='/data/mshukor/data', tokenizer=None):

    sep_token = config.get('sep_token', True)
    randkw_p = config.get('randkw_p', None)


    num_kws = config.get('num_kws', 15)

    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    pretrain_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])    
    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   
    
    if dataset=='pretrain':
        dataset = pretrain_kw_img_dataset(config['train_file'], pretrain_transform, data_dir=data_dir, tokenizer=tokenizer, 
            sep_token=sep_token, randkw_p=randkw_p, num_kws=num_kws)                  
        return dataset      
               
    elif dataset=='re':          
        train_dataset = re_kw_img_train_dataset(config['train_file'], train_transform, config['image_root'], tokenizer=tokenizer, 
            sep_token=sep_token, randkw_p=randkw_p, num_kws=num_kws)
        val_dataset = re_kw_img_eval_dataset(config['val_file'], test_transform, config['image_root'], tokenizer=tokenizer, 
            sep_token=sep_token, num_kws=num_kws)  
        test_dataset = re_kw_img_eval_dataset(config['test_file'], test_transform, config['image_root'], tokenizer=tokenizer, 
            sep_token=sep_token, num_kws=num_kws)                
        return train_dataset, val_dataset, test_dataset   

    elif dataset=='vqa': 
        train_dataset = vqa_kw_img_dataset(config['train_file'], train_transform, config['vqa_root'], config['vg_root'], split='train', tokenizer=tokenizer, 
            sep_token=sep_token, randkw_p=randkw_p, num_kws=num_kws) 
        vqa_test_dataset = vqa_kw_img_dataset(config['test_file'], test_transform, config['vqa_root'], config['vg_root'], split='test', answer_list=config['answer_list'], tokenizer=tokenizer, 
            sep_token=sep_token, num_kws=num_kws)       
        return train_dataset, vqa_test_dataset

    elif dataset=='nlvr':   
        train_dataset = nlvr_kw_img_dataset(config['train_file'], train_transform, config['image_root'], sep_token=sep_token, randkw_p=randkw_p)  
        val_dataset = nlvr_kw_img_dataset(config['val_file'], test_transform, config['image_root'], sep_token=sep_token)  
        test_dataset = nlvr_kw_img_dataset(config['test_file'], test_transform, config['image_root'], sep_token=sep_token)                
        return train_dataset, val_dataset, test_dataset        
               
    elif dataset=='ve':   
        train_dataset = ve_kw_img_dataset(config['train_file'], train_transform, config['image_root'], sep_token=sep_token, randkw_p=randkw_p)  
        val_dataset = ve_kw_img_dataset(config['val_file'], test_transform, config['image_root'], sep_token=sep_token)  
        test_dataset = ve_kw_img_dataset(config['test_file'], test_transform, config['image_root'], sep_token=sep_token)                
        return train_dataset, val_dataset, test_dataset     
    
    elif dataset=='grounding':
        train_transform = transforms.Compose([                        
                transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])         
        train_dataset = grounding_kw_img_dataset(config['train_file'], train_transform, config['image_root'], mode='train', sep_token=sep_token, randkw_p=randkw_p)       
        test_dataset = grounding_kw_img_dataset(config['test_file'], test_transform, config['image_root'], mode='test', sep_token=sep_token)             
        return train_dataset, test_dataset    



    
def create_dataset(dataset, config, data_dir='/data/mshukor/data'):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    pretrain_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])    
    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   
    
    if dataset=='pretrain':
        dataset = pretrain_dataset(config['train_file'], pretrain_transform, data_dir=data_dir)                  
        return dataset      
               
    elif dataset=='re':          
        train_dataset = re_train_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset   

    elif dataset=='vqa': 
        train_dataset = vqa_dataset(config['train_file'], train_transform, config['vqa_root'], config['vg_root'], split='train') 
        vqa_test_dataset = vqa_dataset(config['test_file'], test_transform, config['vqa_root'], config['vg_root'], split='test', answer_list=config['answer_list'])       
        return train_dataset, vqa_test_dataset

    elif dataset=='nlvr':   
        train_dataset = nlvr_dataset(config['train_file'], train_transform, config['image_root'])  
        val_dataset = nlvr_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = nlvr_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset        
               
    elif dataset=='ve':   
        train_dataset = ve_dataset(config['train_file'], train_transform, config['image_root'])  
        val_dataset = ve_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = ve_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset     
    
    elif dataset=='grounding':
        train_transform = transforms.Compose([                        
                transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])         
        train_dataset = grounding_dataset(config['train_file'], train_transform, config['image_root'], mode='train')       
        test_dataset = grounding_dataset(config['test_file'], test_transform, config['image_root'], mode='test')             
        return train_dataset, test_dataset    
    

def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n


def vqa_caption_kw_fuse_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    title_list, kwords_list = [], []

    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question[0])
        kwords_list.append(question[1])
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), (question_list, kwords_list), answer_list, torch.Tensor(weight_list), n


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    