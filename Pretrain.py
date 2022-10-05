'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
 
import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_pretrain import ViCHA
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader, create_kw_img_dataset
from scheduler import create_scheduler
from optim import create_optimizer


from models.model_pretrain_mae import MAE_ViCHA
from models.model_pretrain_kw_img import kw_img_ViCHA





def count_parameters(model, requires_grad=True, avoided_params=None):
    sum_param = 0
    for n, p in model.named_parameters():
        if (p.requires_grad or not requires_grad):
            if avoided_params is not None:
                if not any(t in n for t in avoided_params):
                    sum_param += p.numel()
            else:
                sum_param += p.numel()

    return sum_param


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    mae = config.get('mae', False)
    if config.get('model_mae', False) or mae:
        metric_logger.add_meter('loss_mae', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  

    model_mae = config.get('model_mae', False)
    
    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        optimizer.zero_grad()
  
        image = image.to(device,non_blocking=True) 
        if config.get('kw_img_dataset', False):
            caption = tokenizer(text[0], padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device) 
            kw = tokenizer(text[1], padding='longest', truncation=True, max_length=55, return_tensors="pt").to(device) 
            text_input = [caption, kw]
        else:
            text_input = tokenizer(text, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device)  
        
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader)) 
        
        if model_mae or mae:
            loss_mlm, loss_ita, loss_itm, loss_mae = model(image, text_input, alpha = alpha)  
            loss = loss_mlm + loss_ita + loss_itm + loss_mae
            metric_logger.update(loss_mae=loss_mae.item())
        else:
            loss_mlm, loss_ita, loss_itm = model(image, text_input, alpha = alpha)  
            loss = loss_mlm + loss_ita + loss_itm
                
                
              
        loss.backward()

        optimizer.step()    
        
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])         
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    
    
    
def main(args, config):

    utils.init_distributed_mode(args)    

    
    device = torch.device(args.device)
    print(config)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']    

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder, local_files_only=True)

    save_step = config.get('save_step', None)

    #### Dataset #### 
    print("Creating dataset")
    train_files = []
    for p in config['train_file']:
        train_files.append(os.path.join(args.data_json_dir, p))
        
    config['train_file'] = train_files


    if config.get('kw_img_dataset', False):
        print("loading kw_img_dataset...")
        datasets = [create_kw_img_dataset('pretrain', config, data_dir=args.data_dir)]
    else:
        datasets = [create_dataset('pretrain', config, data_dir=args.data_dir)]
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True], num_tasks, global_rank)         
    else:
        samplers = [None]

    data_loader = create_loader(datasets,samplers,batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]


    #### Model #### 
    print("Creating model")
    if config.get('kw_img_model', False):
        print('loading kw_img_model ...')
        model = kw_img_ViCHA(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)
        find_unused_parameters = False
    elif config.get('model_mae', False):
        print('loading model_mae ...')
        model = MAE_ViCHA(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)
        find_unused_parameters = False
    else:
        model = ViCHA(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)
        find_unused_parameters=False
    
    model = model.to(device)   
        
    print("model stats ...")
    num_param_train = count_parameters(model, requires_grad=False, avoided_params=None)
    avoided_params = ['decoder_embed', 'mae', 'decoder_pos_embed', 'decoder_blocks', 'decoder_norm', 
    'mask_token', 'vision_proj_hidden', 'text_proj_hidden', ] # avoid mae and ms ita
    num_param_test = count_parameters(model, requires_grad=True, avoided_params=avoided_params) # avoide teacher
    
    print("Number of Parameters Train:", num_param_train)
    print("Number of Parameters Test:", num_param_test)

 
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  

    
    if args.checkpoint:    
        avoid_optim = config.get('avoid_optim', False)
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']                       
        if args.resume:
            if not avoid_optim:
                optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch']+1         
        else:
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)   
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)  
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped       
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped               
        msg = model.load_state_dict(state_dict, strict=False)    
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=find_unused_parameters)
        model_without_ddp = model.module    
    
    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):
        
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)  
            
        train_stats = train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config) 
        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                     
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            if save_step is not None:
                if (epoch+1) % int(save_step) == 0:
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_last.pth'))  
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()  
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--data_json_dir', default='/data/mshukor/data/our_albef_data/json_pretrain')   
    parser.add_argument('--data_dir', default='/data/mshukor/data')   

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)