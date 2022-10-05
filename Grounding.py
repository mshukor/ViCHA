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
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.model_retrieval import ViCHA
from models.model_retrieval_kw_img import kw_img_ViCHA

from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader, create_kw_img_dataset
from dataset.utils import collect_result, grounding_eval
from scheduler import create_scheduler
from optim import create_optimizer

from refTools.refer_python3 import REFER

from pdb import set_trace as breakpoint

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    
    for i,(image, text, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device,non_blocking=True)   
        idx = idx.to(device,non_blocking=True)   

        if config.get('kw_img_dataset', False):
            caption = tokenizer(text[0], padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device) 
            kw = tokenizer(text[1], padding='longest', truncation=True, max_length=55, return_tensors="pt").to(device) 
            text_input = [caption, kw]
        else:
            text_input = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(device)  
  
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))
            
        loss_ita, loss_itm = model(image, text_input,alpha=alpha, idx=idx)                  
        loss = loss_ita + loss_itm
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


def val_kw_img(model, data_loader, tokenizer, device, gradcam_mode, block_num):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 50
    feat_map_res = 23 ## 24 16
    if gradcam_mode=='itm':
        model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True
     
    result = []
    for image, text, ref_ids in metric_logger.log_every(data_loader, print_freq, header):
        image, kwords_ = image
        image = image.to(device)

        text_input = tokenizer(text, padding='longest', return_tensors="pt").to(device)  
        
        kwords = tokenizer(kwords_, padding='max_length', truncation=True, max_length=55, return_tensors="pt").to(device) 

        ### kw
       
        kw_output = model.kw_encoder(kwords.input_ids, attention_mask = kwords.attention_mask,                      
                                        return_dict = True, mode = 'text')  
        kw_embeds = kw_output.last_hidden_state
        kw_embeds = model.kw_proj(kw_embeds)


        kw_embeds_external = kw_embeds

        if gradcam_mode=='itm':

            image_feat = model.visual_encoder(image, external_features=kw_embeds_external)     

            image_atts_before = torch.ones(image_feat.size()[:-1],dtype=torch.long).to(image.device)

            image_atts = image_atts_before


            image_embeds =  image_feat        
  


            output = model.text_encoder(text_input.input_ids, 
                                    attention_mask = text_input.attention_mask,
                                    encoder_hidden_states = image_embeds,
                                    encoder_attention_mask = image_atts,      
                                    return_dict = True,
                                   )     

            vl_embeddings = output.last_hidden_state[:,0,:]
            vl_output = model.itm_head(vl_embeddings)   
            loss = vl_output[:,1].sum()   
            
            model.zero_grad()
            loss.backward()    

            with torch.no_grad():                                 
                mask = text_input.attention_mask.view(text_input.attention_mask.size(0),1,-1,1,1)
                
                grads = model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attn_gradients().detach()
                cams = model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attention_map().detach()

                cams = cams[:, :, :, 1:-kw_embeds.shape[1]].reshape(image.size(0), 12, -1, feat_map_res, feat_map_res) * mask              
                grads = grads[:, :, :, 1:-kw_embeds.shape[1]].clamp(min=0).reshape(image.size(0), 12, -1, feat_map_res, feat_map_res) * mask
                
                gradcam = cams * grads
                gradcam = gradcam.mean(1).mean(1)

        elif gradcam_mode=='itc':   
            image_feat = model.visual_encoder(image, external_features=kw_embeds_external, register_blk=block_num)     

            image_atts_before = torch.ones(image_feat.size()[:-1],dtype=torch.long).to(image.device)

            image_atts = image_atts_before


            image_embeds =  image_feat   


            if model.kw_img_token:
                image_embeds_token = torch.cat([image_feat[:,0,:].unsqueeze(1), image_feat[:,-kw_embeds.shape[1],:].unsqueeze(1)], dim=1).mean(1)
            else:
                image_embeds_token = image_feat[:,0,:]

            image_feat = model.vision_proj(image_embeds_token)            
            image_feat = F.normalize(image_feat,dim=-1) 

            text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask,                 
                                             return_dict = True, mode = 'text')            
            text_embeds = text_output.last_hidden_state
            text_feat = F.normalize(model.text_proj(text_embeds[:,0,:]),dim=-1)     
            sim = image_feat@text_feat.t()/model.temp
            loss = sim.diag().sum()
            
            model.zero_grad()
            loss.backward()    

            with torch.no_grad():
                grad = model.visual_encoder.blocks[block_num].attn.get_attn_gradients().detach()
                cam = model.visual_encoder.blocks[block_num].attn.get_attention_map().detach()
                cam = cam[:, :, 0, 1:-kw_embeds.shape[1]].reshape(image.size(0), -1, feat_map_res, feat_map_res)
                grad = grad[:, :, 0, 1:-kw_embeds.shape[1]].reshape(image.size(0), -1, feat_map_res, feat_map_res).clamp(0)
                gradcam = (cam * grad).mean(1)

        for r_id, cam in zip(ref_ids, gradcam):
            result.append({'ref_id':r_id.item(), 'pred':cam})
  
    if gradcam_mode=='itm':
        model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = False             
    return result



def val(model, data_loader, tokenizer, device, gradcam_mode, block_num):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 50
    
    if gradcam_mode=='itm':
        model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True
    
    feat_map_res = 23 ## 24
    result = []
    for image, text, ref_ids in metric_logger.log_every(data_loader, print_freq, header):
        image = image.to(device)
        text_input = tokenizer(text, padding='longest', return_tensors="pt").to(device)  
        
        if gradcam_mode=='itm':
            image_embeds = model.visual_encoder(image) 
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
            output = model.text_encoder(text_input.input_ids, 
                                    attention_mask = text_input.attention_mask,
                                    encoder_hidden_states = image_embeds,
                                    encoder_attention_mask = image_atts,      
                                    return_dict = True,
                                   )     

            vl_embeddings = output.last_hidden_state[:,0,:]
            vl_output = model.itm_head(vl_embeddings)   
            loss = vl_output[:,1].sum()   
            
            model.zero_grad()
            loss.backward()    
 
            with torch.no_grad():                                 
                mask = text_input.attention_mask.view(text_input.attention_mask.size(0),1,-1,1,1)
                
                grads = model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attn_gradients().detach()
                cams = model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attention_map().detach()

                cams = cams[:, :, :, 1:].reshape(image.size(0), 12, -1, feat_map_res, feat_map_res) * mask              
                grads = grads[:, :, :, 1:].clamp(min=0).reshape(image.size(0), 12, -1, feat_map_res, feat_map_res) * mask
                
                gradcam = cams * grads
                gradcam = gradcam.mean(1).mean(1)

        elif gradcam_mode=='itc':    
            image_embeds = model.visual_encoder(image, register_blk=block_num) 
            image_feat = F.normalize(model.vision_proj(image_embeds[:,0,:]),dim=-1) 
            text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask,                 
                                             return_dict = True, mode = 'text')            
            text_embeds = text_output.last_hidden_state
            text_feat = F.normalize(model.text_proj(text_embeds[:,0,:]),dim=-1)     
            sim = image_feat@text_feat.t()/model.temp
            loss = sim.diag().sum()
            
            model.zero_grad()
            loss.backward()    

            with torch.no_grad():
                grad = model.visual_encoder.blocks[block_num].attn.get_attn_gradients().detach()
                cam = model.visual_encoder.blocks[block_num].attn.get_attention_map().detach()
                cam = cam[:, :, 0, 1:].reshape(image.size(0), -1, feat_map_res, feat_map_res)
                grad = grad[:, :, 0, 1:].reshape(image.size(0), -1, feat_map_res, feat_map_res).clamp(0)
                gradcam = (cam * grad).mean(1)

        for r_id, cam in zip(ref_ids, gradcam):
            result.append({'ref_id':r_id.item(), 'pred':cam})
  
    if gradcam_mode=='itm':
        model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = False             
    return result


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")

    train_files = []
    for p in config['train_file']:
        train_files.append(os.path.join(args.data_json_dir, p))
    config['train_file'] = train_files

    test_files = []
    for p in config['test_file']:
        test_files.append(os.path.join(args.data_json_dir, p))
    config['test_file'] = test_files

    config['det_file'] = os.path.join(args.data_json_dir, config['det_file'])
    config['coco_file'] = os.path.join(args.data_json_dir, config['coco_file'])

    config['image_root'] = os.path.join(args.data_dir, config['image_root'])

    config['refcoco_data'] = args.data_json_dir
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder, local_files_only=True)
    
    if config.get('kw_img_dataset', False):
        print('loading kw_img_dataset ...')
        grd_train_dataset, grd_test_dataset  = create_kw_img_dataset('grounding', config)  
    else:
        grd_train_dataset, grd_test_dataset = create_dataset('grounding', config) 

    datasets = [grd_train_dataset, grd_test_dataset]
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)         
    else:
        samplers = [None, None]

    train_loader, test_loader = create_loader(datasets,samplers,batch_size=[config['batch_size'],config['batch_size']],
                                              num_workers=[4,4],is_trains=[True, False], collate_fns=[None,None])
       
    
        
    ## refcoco evaluation tools
    refer = REFER(config['refcoco_data'], 'refcoco+', 'unc')
    dets = json.load(open(config['det_file'],'r'))
    cocos = json.load(open(config['coco_file'],'r'))    

    #### Model #### 
    print("Creating model")

    if config.get('kw_img_model', False):
        print('loading kw_img_model ...')
        model = kw_img_ViCHA(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
        find_unused_parameters = False
    else:
        model = ViCHA(config = config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 
        
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.','')         
                state_dict[encoder_key] = state_dict[key] 
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

    feat_map_res = 23 # 24 16
    print("Start training")
    start_time = time.time()    
    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)  
            
        if config.get('kw_img_dataset', False):
            print('evaluate kw_img_dataset')
            result = val_kw_img(model_without_ddp, test_loader, tokenizer, device, args.gradcam_mode, args.block_num)
        else:
            result = val(model_without_ddp, test_loader, tokenizer, device, args.gradcam_mode, args.block_num)

        results = collect_result(result, args.result_dir, 'epoch%d'%epoch, is_json=False, is_list=True)

        if utils.is_main_process():  
            
            grounding_acc = grounding_eval(results, dets, cocos, refer, alpha=0.5, mask_size=feat_map_res)
            
            if args.evaluate:      
                log_stats = {**{f'{k}': v for k, v in grounding_acc.items()},
                             'epoch': epoch,
                            }                   
            else:             
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'{k}': v for k, v in grounding_acc.items()},
                             'epoch': epoch,
                            }      
                if grounding_acc['val_d']>best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  
                    best = grounding_acc['val_d']
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n") 
                
        if args.evaluate: 
            break                
        
        lr_scheduler.step(epoch+warmup_steps+1)  
        dist.barrier()   

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Grounding.yaml')
    parser.add_argument('--checkpoint', default='')   
    parser.add_argument('--output_dir', default='output/RefCOCO')   
    parser.add_argument('--gradcam_mode', default='itm', choices=['itm','itc']) 
    parser.add_argument('--block_num', default=8, type=int)
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

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)