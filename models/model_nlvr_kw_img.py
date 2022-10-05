from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel

import torch
from torch import nn
import torch.nn.functional as F

class kw_img_ViCHA(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.distill = config['distill']

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    

        bert_config = BertConfig.from_json_file(config['bert_config'])
        bert_config.num_hidden_layers = 18
        
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)      
        self.cls_head = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, 2)
                )            

        self.share_cross_attention(self.text_encoder.encoder)

        ####### kw encoder


        bert_config_kw = BertConfig.from_json_file(config['bert_config'])
        bert_config_kw.num_hidden_layers = config.get('num_hidden_layers_kw', 2)

        bert_config_kw.fusion_layer = config.get('num_hidden_layers_kw', 2)

        text_width = self.text_encoder.config.hidden_size
        vision_width = config['vision_width']
        self.kw_encoder = BertModel.from_pretrained(text_encoder, config=bert_config_kw, add_pooling_layer=False)      
        self.kw_proj = nn.Linear(text_width, vision_width)


        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))                 
            self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False) 
            self.share_cross_attention(self.text_encoder_m.encoder)                

            self.cls_head_m = nn.Sequential(
                      nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                      nn.ReLU(),
                      nn.Linear(self.text_encoder.config.hidden_size, 2)
                    )                

            ####### kw encoder
            self.kw_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config_kw, add_pooling_layer=False)      
            self.kw_proj_m = nn.Linear(text_width, vision_width)

            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.cls_head,self.cls_head_m],
                                [self.kw_encoder,self.kw_encoder_m],
                                [self.kw_proj,self.kw_proj_m],
                               ]
            self.copy_params()        
            self.momentum = 0.995
            


    def forward(self, image, text, targets, alpha=0, train=True):
        
        text, kwords1, kwords2 = text
        ### kw
        kw_output1 = self.kw_encoder(kwords1.input_ids, attention_mask = kwords1.attention_mask,                      
                                        return_dict = True, mode = 'text')  
        kw_embeds1 = kw_output1.last_hidden_state
        kw_embeds1 = self.kw_proj(kw_embeds1)

        kw_output2 = self.kw_encoder(kwords2.input_ids, attention_mask = kwords2.attention_mask,                      
                                        return_dict = True, mode = 'text')  
        kw_embeds2 = kw_output2.last_hidden_state
        kw_embeds2 = self.kw_proj(kw_embeds2)


        kw_embeds_external1 = kw_embeds1
        kw_embeds_external2 = kw_embeds2



        image0, image1 = torch.split(image,targets.size(0)) 

        image0_embeds = self.visual_encoder(image0, external_features=kw_embeds_external1) 
        image_atts_before0 = torch.ones(image0_embeds.size()[:-1],dtype=torch.long).to(image0.device)

        image1_embeds = self.visual_encoder(image1, external_features=kw_embeds_external2) 
        image_atts_before1 = torch.ones(image1_embeds.size()[:-1],dtype=torch.long).to(image1.device)
        



        image0_atts = image_atts_before0
        image1_atts = image_atts_before1



        output = self.text_encoder(text.input_ids, 
                                   attention_mask = text.attention_mask, 
                                   encoder_hidden_states = [image0_embeds,image1_embeds],
                                   encoder_attention_mask = [image0_atts, image1_atts],        
                                   return_dict = True,
                                  )  
        hidden_state = output.last_hidden_state[:,0,:]            
        prediction = self.cls_head(hidden_state)

        if train:
            if self.distill:                
                with torch.no_grad():
                    self._momentum_update()
                    ## kw
                    kw_output1_m = self.kw_encoder_m(kwords1.input_ids, attention_mask = kwords1.attention_mask,                      
                                                    return_dict = True, mode = 'text')  
                    kw_embeds1_m = kw_output1_m.last_hidden_state
                    kw_embeds1_m = self.kw_proj_m(kw_embeds1_m)

                    kw_output2_m = self.kw_encoder_m(kwords2.input_ids, attention_mask = kwords2.attention_mask,                      
                                                    return_dict = True, mode = 'text')  
                    kw_embeds2_m = kw_output2_m.last_hidden_state
                    kw_embeds2_m = self.kw_proj_m(kw_embeds2_m)


                    kw_embeds_external1_m = kw_embeds1_m
                    kw_embeds_external2_m = kw_embeds2_m

                    image0_embeds_m = self.visual_encoder(image0, external_features=kw_embeds_external1_m) 

                    image1_embeds_m = self.visual_encoder(image1, external_features=kw_embeds_external2_m) 




                    output_m = self.text_encoder_m(text.input_ids, 
                                               attention_mask = text.attention_mask, 
                                               encoder_hidden_states = [image0_embeds_m,image1_embeds_m],
                                               encoder_attention_mask = [image0_atts, image1_atts],        
                                               return_dict = True,
                                              )    
                    prediction_m = self.cls_head_m(output_m.last_hidden_state[:,0,:])   

                loss = (1-alpha)*F.cross_entropy(prediction, targets) - alpha*torch.sum(
                    F.log_softmax(prediction, dim=1)*F.softmax(prediction_m, dim=1),dim=1).mean()                        
            else:        
                loss = F.cross_entropy(prediction, targets)     
            return loss  
        else:
            return prediction
 


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
                

    def share_cross_attention(self, model):
            
        for i in range(6):
            layer_num = 6+i*2
            modules_0 = model.layer[layer_num].crossattention.self._modules
            modules_1 = model.layer[layer_num+1].crossattention.self._modules

            for name in modules_0.keys():
                if 'key' in name or 'value' in name:
                    module_0 = modules_0[name]
                    module_1 = modules_1[name]
                    if hasattr(module_0, "weight"):
                        module_0.weight = module_1.weight
                        if hasattr(module_0, "bias"):
                            module_0.bias = module_1.bias    