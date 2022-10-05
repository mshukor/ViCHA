import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
 
from models.model_ve import ViCHA
from models.model_ve_kw_img import kw_img_ViCHA

from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader, create_kw_img_dataset
from scheduler import create_scheduler
from optim import create_optimizer




def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    print('len data loader', len(data_loader))
    for i,(images, text, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    
        images, targets = images.to(device,non_blocking=True), targets.to(device,non_blocking=True)

        if config.get('kw_img_dataset', False):
            caption = tokenizer(text[0], padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device) 
            kw = tokenizer(text[1], padding='longest', truncation=True, max_length=55, return_tensors="pt").to(device) 
            text_inputs = [caption, kw]
        else:
            text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device) 
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        loss = model(images, text_inputs, targets=targets, train=True, alpha=alpha)    
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
               
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, config):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    for images, text, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        images, targets = images.to(device,non_blocking=True), targets.to(device,non_blocking=True)   
        
        if config.get('kw_img_dataset', False):
            caption = tokenizer(text[0], padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device) 
            kw = tokenizer(text[1], padding='longest', truncation=True, max_length=55, return_tensors="pt").to(device) 
            text_inputs = [caption, kw]
        else:
            text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device) 

        prediction = model(images, text_inputs, targets=targets, train=False)  
 
        _, pred_class = prediction.max(1)
        accuracy = (targets==pred_class).sum() / targets.size(0)

        metric_logger.meters['acc'].update(accuracy.item(), n=images.size(0))
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())   
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    
    
def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    print(config)
    #### Dataset #### 
    print("Creating dataset")

    config['train_file'] = os.path.join(args.data_json_dir, config['train_file'])
    config['val_file'] = os.path.join(args.data_json_dir, config['val_file'])
    config['test_file'] = os.path.join(args.data_json_dir, config['test_file'])

    config['image_root'] = os.path.join(args.data_dir, config['image_root'])


    if config.get('kw_img_dataset', False):
        print("loading kw_img_dataset...")
        datasets = create_kw_img_dataset('ve', config, data_dir=args.data_dir)
    else:
        datasets = create_dataset('ve', config) 

    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(datasets,samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],is_trains=[True,False,False], 
                                                          collate_fns=[None,None,None])

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder, local_files_only=True)

    #### Model #### 

    print("Creating model")
    if config.get('kw_img_model', False):
        print('loading kw_img_model ...')
        model = kw_img_ViCHA(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    else:
        model = ViCHA(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        
        if not args.evaluate:
            if config['distill']:
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
                state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 

            for key in list(state_dict.keys()):                
                if 'bert' in key:
                    new_key = key.replace('bert.','')
                    state_dict[new_key] = state_dict[key] 
                    del state_dict[key]
                
        msg = model.load_state_dict(state_dict,strict=False)
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0
    
    print("Start training")
    start_time = time.time()

    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)  
            
        val_stats = evaluate(model, val_loader, tokenizer, device, config)
        test_stats = evaluate(model, test_loader, tokenizer, device, config)

        if utils.is_main_process():  
            if args.evaluate:
                log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                            }

                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")                
            else:    
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                            }

                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if float(val_stats['acc'])>best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 
                    best = float(val_stats['acc'])
                    best_epoch = epoch
        
        if args.evaluate:
            break
        lr_scheduler.step(epoch+warmup_steps+1)  
        dist.barrier()   
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)         
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VE.yaml')
    parser.add_argument('--output_dir', default='output/VE')  
    parser.add_argument('--checkpoint', default='')   
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--data_json_dir', default='/data/mshukor/data/our_albef_data/data')   
    parser.add_argument('--data_dir', default='/data/mshukor/data')  
 
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
