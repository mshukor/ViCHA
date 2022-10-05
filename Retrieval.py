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
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
  
import utils
from dataset import create_dataset, create_sampler, create_loader, create_kw_img_dataset
from scheduler import create_scheduler
from optim import create_optimizer

from tqdm import tqdm 

from models.model_retrieval_kw_img import kw_img_ViCHA



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




@torch.no_grad()
def evaluation_kw_img(model, data_loader, tokenizer, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 64
    text_feats = []
    text_embeds = []  
    text_atts = []
    for i in tqdm(range(0, num_text, text_bs)):
        text = texts[i: min(num_text, i+text_bs)]

        caption = [t[0] for t in text]

        text_input = tokenizer(caption, padding='max_length', truncation=True, max_length=25, return_tensors="pt").to(device) 

        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:,0,:]))
        text_embeds.append(text_embed)   
        text_feats.append(text_feat.cpu())
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds,dim=0)
    text_feats = torch.cat(text_feats,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    
    image_feats = []
    image_embeds = []
    image_attss = []
    for image, img_id in tqdm(data_loader): 

        image, kwords_ = image

        image = image.to(device) 

        kwords = tokenizer(kwords_, padding='max_length', truncation=True, max_length=55, return_tensors="pt").to(device) 

        ### kw
        if not model.kw_img_fusion:
            if model.num_hidden_layers_kw == 0:
                input_shape = kwords.input_ids.size()
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=image.device)

                kw_embeds = model.kw_encoder(input_ids=kwords.input_ids,
                position_ids=None,
                token_type_ids=token_type_ids,
                inputs_embeds=None,
                past_key_values_length=0,)
            else:
                kw_output = model.kw_encoder(kwords.input_ids, attention_mask = kwords.attention_mask,                      
                                                return_dict = True, mode = 'text')  
                kw_embeds = kw_output.last_hidden_state
            kw_embeds = model.kw_proj(kw_embeds)

        if model.late_cat:
            kw_embeds_external = None
        else:
            kw_embeds_external = kw_embeds



        image_feat = model.visual_encoder(image, external_features=kw_embeds_external)     

        image_atts_before = torch.ones(image_feat.size()[:-1],dtype=torch.long).to(image.device)

        if model.kw_img_fusion:
            kw_output = model.kw_encoder(kwords.input_ids, attention_mask = kwords.attention_mask, 
                encoder_hidden_states = image_feat, encoder_attention_mask = image_atts_before,                         
                                            return_dict = True, mode = 'fusion') 
            kw_embeds = kw_output.last_hidden_state
            kw_embeds = model.kw_proj(kw_embeds)


        if model.late_cat:
            if model.kw_only:
                image_feat = kw_embeds
                image_atts = kwords.attention_mask
            else:
                image_feat = torch.cat((image_feat, kw_embeds), dim=1)
                image_atts = torch.cat((image_atts_before, kwords.attention_mask), dim=1)  
        else:
            image_atts = image_atts_before



        if model.kw_img_token:
            image_embeds_token = torch.cat([image_feat[:,0,:].unsqueeze(1), image_feat[:,len(image_atts_before),:].unsqueeze(1)], dim=1).mean(1)
        else:
            image_embeds_token = image_feat[:,0,:]

        image_embed = model.vision_proj(image_embeds_token)            
        image_embed = F.normalize(image_embed,dim=-1)      
        
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)
        image_attss.append(image_atts)
     
    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)
    image_attss = torch.cat(image_attss,dim=0)
    
    sims_matrix = image_embeds @ text_embeds.t()
    dual_metrics= config.get('dual_metrics', False)
    if dual_metrics:
        return sims_matrix.cpu().numpy(), sims_matrix.t().cpu().numpy()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image),len(texts)),-100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_feats[start+i].repeat(config['k_test'],1,1)
        encoder_att = image_attss[start+i].repeat(config['k_test'],1)

        # encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)

        output = model.text_encoder(encoder_embeds = text_feats[topk_idx].to(device), 
                                    attention_mask = text_atts[topk_idx].to(device) ,
                                    encoder_hidden_states = encoder_output.to(device) ,
                                    encoder_attention_mask = encoder_att.to(device) ,                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_i2t[start+i,topk_idx] = score
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    
    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx]
        encoder_att = image_attss[topk_idx]

        # encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds = text_feats[start+i].repeat(config['k_test'],1,1).to(device) , 
                                    attention_mask = text_atts[start+i].repeat(config['k_test'],1).to(device) ,
                                    encoder_hidden_states = encoder_output.to(device) ,
                                    encoder_attention_mask = encoder_att.to(device) ,                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i,topk_idx] = score

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()



@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 64
    text_feats = []
    text_embeds = []  
    text_atts = []
    for i in tqdm(range(0, num_text, text_bs)):
        text = texts[i: min(num_text, i+text_bs)]
        
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=25, return_tensors="pt").to(device) 


        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:,0,:]))
        text_embeds.append(text_embed)   
        text_feats.append(text_feat.cpu())
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds,dim=0)
    text_feats = torch.cat(text_feats,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    
    image_feats = []
    image_embeds = []
    for image, img_id in tqdm(data_loader): 
        image = image.to(device) 
        if model.visual_encoder.return_hidden_state:
            image_feat, _ = model.visual_encoder(image)  
        else: 
            image_feat = model.visual_encoder(image)  

        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed,dim=-1)      
        
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)
     
    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)
    
    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image),len(texts)),-100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_feats[start+i].repeat(config['k_test'],1,1)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds = text_feats[topk_idx].to(device), 
                                    attention_mask = text_atts[topk_idx].to(device) ,
                                    encoder_hidden_states = encoder_output.to(device) ,
                                    encoder_attention_mask = encoder_att.to(device) ,                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_i2t[start+i,topk_idx] = score
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    
    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds = text_feats[start+i].repeat(config['k_test'],1,1).to(device) , 
                                    attention_mask = text_atts[start+i].repeat(config['k_test'],1).to(device) ,
                                    encoder_hidden_states = encoder_output.to(device) ,
                                    encoder_attention_mask = encoder_att.to(device) ,                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i,topk_idx] = score

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


            
@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result




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
    print("Creating retrieval dataset")

    
    print("Creating dataset")

    config['train_file'] = os.path.join(args.data_json_dir, config['train_file'])
    config['test_file'] = os.path.join(args.data_json_dir, config['test_file'])
    config['val_file'] = os.path.join(args.data_json_dir, config['val_file'])

    config['image_root'] = os.path.join(args.data_dir, config['image_root'])

    print(config)


    tokenizer = BertTokenizer.from_pretrained(args.text_encoder, local_files_only=True)

    if config.get('kw_img_dataset', False):
        print('loading kw_img_dataset ...')
        train_dataset, val_dataset, test_dataset = create_kw_img_dataset('re', config)  
    else:
        train_dataset, val_dataset, test_dataset = create_dataset('re', config)  
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None,None,None])   
       

    #### Model #### 
    print("Creating model")

    if config.get('kw_img_model', False):
        print('loading kw_img_model ...')
        model = kw_img_ViCHA(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
        find_unused_parameters = False
    else:
        model = ViCHA(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
        find_unused_parameters = False

    if args.checkpoint:    
        if args.checkpoint0 is not None:
            checkpoint = torch.load(args.checkpoint0, map_location='cpu') 
            state_dict = checkpoint['model']
            
            # reshape positional embedding to accomodate for image resolution change
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
            
            print('load checkpoint0 from %s'%args.checkpoint0)
            print(msg)  

        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        
        if not config.get('clip_vit', False):
            # reshape positional embedding to accomodate for image resolution change
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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=find_unused_parameters)
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
        
        if config.get('kw_img_dataset', False):
            print('evaluate kw_img_dataset')
            score_val_i2t, score_val_t2i, = evaluation_kw_img(model_without_ddp, val_loader, tokenizer, device, config)
            score_test_i2t, score_test_t2i = evaluation_kw_img(model_without_ddp, test_loader, tokenizer, device, config)
        else:
            score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, tokenizer, device, config)
            score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config)
    
        if utils.is_main_process():  
      
            val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)  
            print(val_result)
            test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)    
            print(test_result)
            
            if args.evaluate:                
                log_stats = {**{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},                  
                             'epoch': epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},                  
                             'epoch': epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")   
                    
                if val_result['r_mean']>best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  
                    best = val_result['r_mean']    
                    best_epoch = epoch
                    
        if args.evaluate: 
            break
           
        lr_scheduler.step(epoch+warmup_steps+1)  
        dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)               

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')        
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
    parser.add_argument('--checkpoint0', default=None)   


    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
