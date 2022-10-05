from create_keywords import compute_sim, dict_to_tensor, filter_topk_per_image_dataset_from_saved



# percentage = 0.5
# image_root = None
# data_ = None
# captions_embed = None 
# json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/vg_ttl_kw.json'

# caption_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/caption_embeddings/vg_caption_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/vg_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/vg_ttl_kw_'+str(percentage)+'.json'



# tmp = filter_topk_per_image_dataset_from_saved(json_path, caption_embeddings_path, image_embeddings_path, 
#                                                per=percentage, output_path=output_path, 
#                                                image_root=image_root, overwrite=True, caption_embed=captions_embed, data_=data_)


# percentage = 0.15
# image_root = None
# data_ = None
# captions_embed = None 
# json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/vg_ttl_kw.json'

# caption_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/caption_embeddings/vg_caption_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/vg_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/vg_ttl_kw_'+str(percentage)+'.json'



# tmp = filter_topk_per_image_dataset_from_saved(json_path, caption_embeddings_path, image_embeddings_path, 
#                                                per=percentage, output_path=output_path, 
#                                                image_root=image_root, overwrite=True, caption_embed=captions_embed, data_=data_)

percentage = 0.7
image_root = None
data_ = None
captions_embed = None 
json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/vg_ttl_kw.json'

caption_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/caption_embeddings/vg_caption_embeddings.json'
image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/vg_image_embeddings.json'

output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/vg_ttl_kw_'+str(percentage)+'.json'



tmp = filter_topk_per_image_dataset_from_saved(json_path, caption_embeddings_path, image_embeddings_path, 
                                               per=percentage, output_path=output_path, 
                                               image_root=image_root, overwrite=True, caption_embed=captions_embed, data_=data_)

###################### ###################### 4OD ######################


# percentage = 0.5
# image_root = None
# data_ = None
# captions_embed = None 
# json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/vg_o4dkw.json'

# caption_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/caption_embeddings/vg_caption_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/vg_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/vg_o4dkw_'+str(percentage)+'.json'



# tmp = filter_topk_per_image_dataset_from_saved(json_path, caption_embeddings_path, image_embeddings_path, 
#                                                per=percentage, output_path=output_path, 
#                                                image_root=image_root, overwrite=True, caption_embed=captions_embed, data_=data_)