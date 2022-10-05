import torch 
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools

from einops import rearrange

  
def mae_kw_img_forward(model, text):
        text, kwords = text
        ### kw
        
        kw_output = model.kw_encoder(kwords.input_ids, attention_mask = kwords.attention_mask,                      
                                        return_dict = True, mode = 'text')  
        kw_embeds = kw_output.last_hidden_state
        kw_embeds = model.kw_proj(kw_embeds)

        return kw_embeds





def compute_mae(model, image, text, mask_ratio=0.75, unimodal=False, uni_cross_modal=False, 
    kw_img=False):
    ## FROM https://github.com/facebookresearch/mae/blob/main/models_mae.py


    def random_masking(x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def patchify(model, imgs, patch_size=16):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        # p = patch_size
        p = model.patch_size
        # p = int((model.visual_encoder.patch_embed.num_patches)**0.5)
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward_encoder(model, x, mask_ratio, register_blk=-1):
        # embed patches
        x = model.visual_encoder.patch_embed(x) # img tp patches (B, N, D)

        x = x + model.visual_encoder.pos_embed[:,1:,:]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = random_masking(x, mask_ratio)

        B = x.shape[0]
        cls_token = model.visual_encoder.cls_token + model.visual_encoder.pos_embed[:, :1, :]
        cls_tokens = model.visual_encoder.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
  
        
        # x = model.pos_drop(x)

        for i, blk in enumerate(model.visual_encoder.blocks):
            x = blk(x, register_blk==i)
        x = model.visual_encoder.norm(x)


        return x, mask, ids_restore

    def forward_encoder_clip(model, x, mask_ratio, register_blk=-1):


        x = model.visual_encoder.clip_model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = x + model.visual_encoder.clip_model.positional_embedding.to(x.dtype).unsqueeze(0)[:,1:,:]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = random_masking(x, mask_ratio)

        B = x.shape[0]

        cls_token = model.visual_encoder.clip_model.class_embedding.to(x.dtype) + model.visual_encoder.clip_model.positional_embedding.to(x.dtype).unsqueeze(0)[:,:1,:]
        cls_tokens = cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)


        x = model.visual_encoder.clip_model.ln_pre(x)


        x = x.permute(1, 0, 2)  # NLD -> LND
        for layer in model.visual_encoder.clip_model.transformer:
            x = layer(x)


        # x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = model.visual_encoder.clip_model.ln_post(x)

        return x, mask, ids_restore


    def forward_loss(model, imgs, pred, mask, norm_pix_loss=True, patch_size=16):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = patchify(model, imgs, patch_size=patch_size)
        if norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5


        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_decoder(model, x, ids_restore):
        # embed tokens
        x = model.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = model.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + model.decoder_pos_embed

        # apply Transformer blocks
        for blk in model.decoder_blocks:
            x = blk(x)
        x = model.decoder_norm(x)

        # predictor projection
        # x = model.decoder_pred(x)

        # remove cls token
        # x = x[:, 1:, :]

        return x

    def forward_mae(model, image, text, mask_ratio=0.75, unimodal=False, uni_cross_modal=False):
        # with torch.no_grad():
        #     model.temp.clamp_(0.001,0.5)
        

        ### image mae
        image_embeds, mask, ids_restore = forward_encoder(model, image, mask_ratio)

        if uni_cross_modal:
            pred_uni = forward_decoder(model, image_embeds, ids_restore)
        elif unimodal:
            pred = forward_decoder(model, image_embeds, ids_restore)
            return pred, mask
        # image_embeds = model.cross_modal_image_transform(image_embeds)

        # append mask tokens to sequence
        mask_tokens = model.mask_token.repeat(image_embeds.shape[0], ids_restore.shape[1] + 1 - image_embeds.shape[1], 1)
        image_embeds_ = torch.cat([image_embeds[:, 1:, :], mask_tokens], dim=1)  # no cls token
        image_embeds_ = torch.gather(image_embeds_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, image_embeds.shape[2]))  # unshuffle
        image_embeds = torch.cat([image_embeds[:, :1, :], image_embeds_], dim=1)  # append cls token


        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)


        text_output = model.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
             

        ###=================================###
        # forward the positve image-text pair reverse the Q and KV
        output_pos = model.text_encoder.bert(encoder_embeds = image_embeds, 
                                        attention_mask = image_atts,
                                        encoder_hidden_states = text_embeds,
                                        encoder_attention_mask = text.attention_mask,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )            
        
        output_pos_embeds = output_pos.last_hidden_state
        if uni_cross_modal:
            return (output_pos_embeds, pred_uni), mask
        else:
            return output_pos_embeds, mask


    infer, mask = forward_mae(model, image, text, mask_ratio=mask_ratio, unimodal=unimodal, 
        uni_cross_modal=uni_cross_modal)

    if uni_cross_modal:
        mae_image_feats = model.mae_proj(infer[0])
        mae_image_feats_uni = model.mae_proj_uni(infer[1])

        loss_mae = forward_loss(model, image, mae_image_feats[:, 1:, :], mask)
        loss_mae_uni = forward_loss(model, image, mae_image_feats_uni[:, 1:, :], mask)
        loss_mae = (loss_mae + loss_mae_uni)/2

    else:
        mae_image_feats = model.mae_proj(infer)
        loss_mae = forward_loss(model, image, mae_image_feats[:, 1:, :], mask)

    
    return loss_mae