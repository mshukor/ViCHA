from create_keywords import filter_topk_dataset_from_saved, filter_topk_dataset


# #################### SBU
# percentage = 0.2
# image_root = None
# json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/sbu_ttl_kw.json'

# caption_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/caption_embeddings/sbu_caption_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/sbu_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/'+str(percentage)+'/sbu_ttl_kw.json'



# tmp = filter_topk_dataset_from_saved(json_path, caption_embeddings_path, image_embeddings_path, per=percentage, output_path=output_path, image_root=image_root, overwrite=True)


# percentage = 0.7
# image_root = None
# json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/sbu_ttl_kw.json'

# caption_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/caption_embeddings/sbu_caption_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/sbu_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/'+str(percentage)+'/sbu_ttl_kw.json'



# tmp = filter_topk_dataset_from_saved(json_path, caption_embeddings_path, image_embeddings_path, per=percentage, output_path=output_path, image_root=image_root, overwrite=True)

# ###################### COCO
# percentage = 0.2
# image_root = None
# json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/coco_ttl_kw.json'

# caption_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/caption_embeddings/coco_caption_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/coco_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/'+str(percentage)+'/coco_ttl_kw.json'



# tmp = filter_topk_dataset_from_saved(json_path, caption_embeddings_path, image_embeddings_path, per=percentage, output_path=output_path, image_root=image_root, overwrite=True)


# ###################### VG

# percentage = 0.2
# image_root = None
# json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/vg_ttl_kw.json'

# caption_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/caption_embeddings/vg_caption_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/vg_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/'+str(percentage)+'/vg_ttl_kw.json'



# tmp = filter_topk_dataset_from_saved(json_path, caption_embeddings_path, image_embeddings_path, per=percentage, output_path=output_path, image_root=image_root, overwrite=True)




###################### CC3M

# percentage = 0.2
# image_root = None
# json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/cc3m_ttl_kw.json'

# caption_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/caption_embeddings/cc3m_caption_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/cc3m_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/'+str(percentage)+'/cc3m_ttl_kw.json'



# tmp = filter_topk_dataset_from_saved(json_path, caption_embeddings_path, image_embeddings_path, per=percentage, output_path=output_path, image_root=image_root, overwrite=True)


# percentage = 0.5
# image_root = None
# json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/cc3m_filter1_ttl_kw.json'

# caption_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/caption_embeddings/cc3m_caption_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/cc3m_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/'+str(percentage)+'/cc3m_filter1_ttl_kw.json'



# tmp = filter_topk_dataset_from_saved(json_path, caption_embeddings_path, image_embeddings_path, per=percentage, output_path=output_path, image_root=image_root, overwrite=True)



# percentage = 0.2
# image_root = None
# json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/cc3m_filter1_ttl_kw.json'

# caption_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/caption_embeddings/cc3m_caption_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/cc3m_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/'+str(percentage)+'/cc3m_filter1_ttl_kw.json'



# tmp = filter_topk_dataset_from_saved(json_path, caption_embeddings_path, image_embeddings_path, per=percentage, output_path=output_path, image_root=image_root, overwrite=True)

percentage = 0.7
image_root = None
json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/cc3m_filter1_ttl_kw.json'


output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/'+str(percentage)+'/cc3m_filter1_ttl_kw.json'
output_path_orig = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/cc3m_filter1_sim_ttl_kw.json'
save_original = True


tmp = filter_topk_dataset(json_path, per=percentage, output_path=output_path, 
	image_root=image_root, overwrite=True, output_path_orig=output_path_orig, 
	save_original=save_original, batch_size=256)

###################### ###################### 4OD ######################

########## SBU
# percentage = 0.5
# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/sbu_o4dkw.json'

# caption_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/caption_embeddings/sbu_caption_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/sbu_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/'+str(percentage)+'/sbu_o4dkw.json'



# tmp = filter_topk_dataset_from_saved(json_path, caption_embeddings_path, image_embeddings_path, per=percentage, output_path=output_path, image_root=image_root, overwrite=True)