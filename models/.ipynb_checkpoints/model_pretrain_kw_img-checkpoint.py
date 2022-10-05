from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed, Block
from models.xbert import BertConfig, BertForMaskedLM, BertModel, BertEmbeddings

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random
  
from models.objectives.mae import compute_mae
from models.pos_embed import get_2d_sincos_pos_embed
 
class kw_img_ViCHA(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,    
                 temp = 0.07,
                 init_deit = True
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']
     
        self.return_hidden_state = config.get('return_hidden_state', False)


        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), return_hidden_state=self.return_hidden_state)   
        
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
            
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)
        text_width = self.text_encoder.config.hidden_size
        ####### kw encoder

        bert_config_kw = BertConfig.from_json_file(config['bert_config'])

        self.num_hidden_layers_kw = config.get('num_hidden_layers_kw', 2)

        if self.num_hidden_layers_kw == 0:
            self.kw_encoder = BertEmbeddings(config=bert_config_kw)   
        else:
            bert_config_kw.num_hidden_layers = self.num_hidden_layers_kw

            bert_config_kw.fusion_layer = config.get('num_hidden_layers_kw', 2)
            self.kw_encoder = BertModel.from_pretrained(text_encoder, config=bert_config_kw, add_pooling_layer=False)  

        self.kw_proj = nn.Linear(text_width, vision_width)



        
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  
        self.itm_head = nn.Linear(text_width, 2)     

        # create momentum models
        self.patch_size = 16
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), return_hidden_state=self.return_hidden_state) 
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)       
        self.text_proj_m = nn.Linear(text_width, embed_dim)    
        

        ####### kw encoder

        if self.num_hidden_layers_kw == 0:
            self.kw_encoder_m = BertEmbeddings(config=bert_config_kw)   
        else:
            self.kw_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config_kw, add_pooling_layer=False)      
        self.kw_proj_m = nn.Linear(text_width, vision_width)

        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                            [self.kw_encoder,self.kw_encoder_m],
                            [self.kw_proj,self.kw_proj_m],
                           ]
        
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        ## ita loss variants
        self.ita_hidden_weight = config.get('ita_hidden_weight', 1)
        self.ita_hidden_weights = config.get('ita_hidden_weights', None)
        self.num_hidden_align = config.get('num_hidden_align', 5)
        if self.return_hidden_state:
            self.vision_proj_hidden = nn.ModuleList([nn.Linear(vision_width, embed_dim) for i in range(self.num_hidden_align)])
            self.text_proj_hidden = nn.ModuleList([nn.Linear(text_width, embed_dim) for i in range(self.num_hidden_align)])


        self.register_blk_id = config.get('register_blk_id', -1)

        self.mae = config.get('mae', False)
        if self.mae:
            self.mae_proj = nn.Linear(vision_width, 16**2 * 3)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, vision_width))
            self.mask_ratio = config.get('mask_ratio', 0.75)

            self.unimodal_mae = config.get('unimodal_mae', False)
            self.uni_cross_modal_mae = config.get('uni_cross_modal_mae', False)

            if self.uni_cross_modal_mae:
                self.mae_proj_uni = nn.Linear(vision_width, 16**2 * 3)

            if self.unimodal_mae or self.uni_cross_modal_mae:
                decoder_embed_dim = 768
                decoder_num_heads = 16
                mlp_ratio = 4
                decoder_depth = 2
                in_chans = 3
                num_patches = int((config['image_res'] // 16) ** 2)

                self.decoder_embed = nn.Linear(vision_width, decoder_embed_dim, bias=True)
                self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

                self.decoder_blocks = nn.ModuleList([
                    Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6))
                    for i in range(decoder_depth)])

                self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
                # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

                ## init
                decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(num_patches**.5), cls_token=True)
                self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
                torch.nn.init.normal_(self.mask_token, std=.02)

                self.mae_proj = nn.Linear(decoder_embed_dim, self.patch_size**2 * 3)
                self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            print('self.uni_cross_modal_mae', self.uni_cross_modal_mae)


    def forward(self, image, text, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        

        text, kwords = text
        ### kw

        if self.num_hidden_layers_kw == 0:
            input_shape = kwords.input_ids.size()
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=image.device)

            kw_embeds = self.kw_encoder(input_ids=kwords.input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,)

        else:
            kw_output = self.kw_encoder(kwords.input_ids, attention_mask = kwords.attention_mask,                      
                                            return_dict = True, mode = 'text') 
            kw_embeds = kw_output.last_hidden_state

        kw_embeds = self.kw_proj(kw_embeds)


        kw_embeds_external = kw_embeds



        if self.return_hidden_state:
            image_embeds, image_hidden_states = self.visual_encoder(image, register_blk=self.register_blk_id, 
                external_features=kw_embeds_external) 
        else:
            image_embeds = self.visual_encoder(image, register_blk=self.register_blk_id, external_features=kw_embeds_external) 
        
        image_atts_before = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

      



        image_atts = image_atts_before



        image_embeds_token = image_embeds[:,0,:]

        image_feat = F.normalize(self.vision_proj(image_embeds_token),dim=-1)  

        text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                 
             
        # get momentum features
        with torch.no_grad():
            self._momentum_update()

            ### kw
            if self.num_hidden_layers_kw == 0:
                kw_embeds_m = self.kw_encoder_m(input_ids=kwords.input_ids,
                position_ids=None,
                token_type_ids=token_type_ids,
                inputs_embeds=None,
                past_key_values_length=0,)
            else:
                kw_output_m = self.kw_encoder_m(kwords.input_ids, attention_mask = kwords.attention_mask,                      
                                                return_dict = True, mode = 'text') 
                kw_embeds_m = kw_output_m.last_hidden_state


            kw_embeds_m = self.kw_proj_m(kw_embeds_m)


            kw_embeds_external_m = kw_embeds_m


            if self.return_hidden_state:
                image_embeds_m, image_hidden_states_m = self.visual_encoder_m(image, external_features=kw_embeds_external_m)
            else:
                image_embeds_m = self.visual_encoder_m(image, external_features=kw_embeds_external_m) 


            image_embeds_token_m = image_embeds_m[:,0,:]


            image_atts_m = image_atts

            ####
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_token_m),dim=-1)  
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)                                         
            text_output_m = self.text_encoder_m.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                                return_dict = True, mode = 'text')    
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp 
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp     

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)          

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

        sim_i2t = image_feat @ text_feat_all / self.temp 
        sim_t2i = text_feat @ image_feat_all / self.temp 
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        if self.return_hidden_state:
            loss_ita_hidden = self.hidden_states_ita(image_hidden_states[-(self.num_hidden_align+1): -1], text_output.hidden_states[-(self.num_hidden_align+1):-1], weights=self.ita_hidden_weights)
            # print('loss_ita', loss_ita)
            loss_ita += loss_ita_hidden*self.ita_hidden_weight


        ###=================================###
        # forward the positve image-text pair
        output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )            
        with torch.no_grad():
            bs = image.size(0)          
            weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)
   
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)     
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)     

        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        output_neg = self.text_encoder.bert(encoder_embeds = text_embeds_all, 
                                        attention_mask = text_atts_all,
                                        encoder_hidden_states = image_embeds_all,
                                        encoder_attention_mask = image_atts_all,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )                         

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)            

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)     
        
        ##================= MLM ========================##                
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
                                      probability_matrix = probability_matrix) 
        
        with torch.no_grad():
            logits_m = self.text_encoder_m(input_ids, 
                                           attention_mask = text.attention_mask,
                                           encoder_hidden_states = image_embeds_m,
                                           encoder_attention_mask = image_atts_m,      
                                           return_dict = True,
                                           return_logits = True,   
                                          )    
        mlm_output = self.text_encoder(input_ids, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                       labels = labels,   
                                       soft_labels = F.softmax(logits_m,dim=-1),
                                       alpha = alpha
                                      )                           
        loss_mlm = mlm_output.loss        
        if self.mae:
            loss_mae = compute_mae(self, image, text, mask_ratio=self.mask_ratio, 
            unimodal=self.unimodal_mae, uni_cross_modal=self.uni_cross_modal_mae)
            return loss_mlm, loss_ita, loss_itm, loss_mae

        return loss_mlm, loss_ita, loss_itm  

        

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
            
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 
        
        
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids
    def hidden_states_ita(self, image_hidden_states, text_hidden_states, weights=None):
        loss_ita_hidden = 0
        if weights is None:
            weights = [1 for i in range(len(image_hidden_states))]
        for i in range(len(image_hidden_states)):
            image_feat_hidden = F.normalize(self.vision_proj_hidden[i](image_hidden_states[i][:,0,:]),dim=-1)  

            text_feat_hidden = F.normalize(self.text_proj_hidden[i](text_hidden_states[i][:,0,:]),dim=-1)                 
                   
            image_feat_hidden_all = concat_all_gather(image_feat_hidden)
            text_feat_hidden_all = concat_all_gather(text_feat_hidden)     

            sim_i2t = image_feat_hidden @ text_feat_hidden_all.t() / self.temp 
            sim_t2i = text_feat_hidden @ image_feat_hidden_all.t() / self.temp 
            
            sim_targets = torch.zeros(sim_i2t.size()).to(sim_i2t.device)
            sim_targets.fill_diagonal_(1)   

            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean() 

            loss_ita_hidden += weights[i]*(loss_i2t+loss_t2i)/2
            # print((loss_i2t+loss_t2i)/2)
        # print(loss_ita_hidden)
        return loss_ita_hidden



@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

