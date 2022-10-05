from create_keywords import dict_to_tensor, select_topk, create_clip_Da_dataset_from_saved, create_titles, create_clip_Da_dataset_from_saved_nlvr

### SBU
# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/json_pretrain/sbu.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/sbu_keywords_embeddings.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/sbu.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/sbu_image_embeddings.json'


# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/sbu.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/sbu_ttl_kw.json'
# tmp2 = create_titles(json_path, output_path=output_path)


### VG
# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/json_pretrain/vg.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/vg_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/vg_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/vg.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)



# json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/vg.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/vg_ttl_kw.json'
# tmp2 = create_titles(json_path, output_path=output_path)



### VQA

# image_root = '/data/mshukor/data/coco'

# json_path = '/data/mshukor/data/our_albef_data/data/vqa_train.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/coco_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/coco_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/vqa_train.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/data/vqa_test.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/coco_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/vqa_test_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/vqa_test.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/data/vqa_val.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/coco_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/vqa_val_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/vqa_val.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# json_path = '/data/mshukor/data/our_albef_data/data/vg_qa.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/vg_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/vg_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/vg_qa.json'

# image_root = '/data/mshukor/data/visual_genome'
# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# title
# json_path = '/data/mshukor/data/our_albef_data/clip_da/data/coco_test.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/coco_test_ttl_kw.json'
# tmp2 = create_titles(json_path, output_path=output_path)

# json_path = '/data/mshukor/data/our_albef_data/clip_da/data/coco_val.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/coco_val_ttl_kw.json'
# tmp2 = create_titles(json_path, output_path=output_path)

# json_path = '/data/mshukor/data/our_albef_data/clip_da/data/coco_train.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/coco_train_ttl_kw.json'
# tmp2 = create_titles(json_path, output_path=output_path)




# json_path = '/data/mshukor/data/our_albef_data/clip_da/data/vqa_train.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/vqa_train_ttl_kw.json'
# tmp2 = create_titles(json_path, output_path=output_path)

# json_path = '/data/mshukor/data/our_albef_data/clip_da/data/vqa_val.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/vqa_val_ttl_kw.json'
# tmp2 = create_titles(json_path, output_path=output_path)

# json_path = '/data/mshukor/data/our_albef_data/clip_da/data/vqa_test.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/vqa_test_ttl_kw.json'
# tmp2 = create_titles(json_path, output_path=output_path)

################# Flicker

# json_path = '/data/mshukor/data/our_albef_data/clip_da/data/flickr30k_test.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/flickr30k_test_ttl_kw.json'
# tmp2 = create_titles(json_path, output_path=output_path)

# json_path = '/data/mshukor/data/our_albef_data/clip_da/data/flickr30k_val.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/flickr30k_val_ttl_kw.json'
# tmp2 = create_titles(json_path, output_path=output_path)

# json_path = '/data/mshukor/data/our_albef_data/clip_da/data/flickr30k_train.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/flickr30k_train_ttl_kw.json'
# tmp2 = create_titles(json_path, output_path=output_path)



#######################3 COCO karpathy

# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/json_pretrain/coco_karp.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/coco_karp_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/coco_karp_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/coco_karp.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)



# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/data/karpathy_coco/coco_test.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/coco_karp_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/coco_test_karp_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/karpathy_coco/coco_test.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/data/karpathy_coco/coco_val.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/coco_karp_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/coco_val_karp_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/karpathy_coco/coco_val.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)



# image_root = '/data/mshukor/data/coco'

# json_path = '/data/mshukor/data/our_albef_data/data/karpathy_coco/coco_train.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/coco_karp_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/coco_karp_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/karpathy_coco/coco_train.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)




# json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/coco_karp.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/coco_karp_ttl_kw.json'
# tmp2 = create_titles(json_path, output_path=output_path)



# json_path = '/data/mshukor/data/our_albef_data/data/karpathy_coco/coco_test.json'
# output_path = '/data/mshukor/data/our_albef_data/data/karpathy_coco/coco_test_ttl_kw.json'
# tmp2 = create_titles(json_path, output_path=output_path)



# json_path = '/data/mshukor/data/our_albef_data/data/karpathy_coco/coco_val.json'
# output_path = '/data/mshukor/data/our_albef_data/data/karpathy_coco/coco_val_ttl_kw.json'
# tmp2 = create_titles(json_path, output_path=output_path)

# json_path = '/data/mshukor/data/our_albef_data/clip_da/data/karpathy_coco/coco_train.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/karpathy_coco/coco_train_ttl_kw.json'
# tmp2 = create_titles(json_path, output_path=output_path)



#### CC3M

# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/json_pretrain/cc3m.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/cc3m_keywords_embeddings.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/cc3m.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/cc3m_image_embeddings.json'


# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/cc3m.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/cc3m_ttl_kw.json'
# tmp2 = create_titles(json_path, output_path=output_path)


## after filtering

# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/json_pretrain/cc3m.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/cc3m_keywords_filter1_embeddings.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/cc3m_filter1.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/cc3m_image_embeddings.json'


# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/cc3m_filter1.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/cc3m_filter1_ttl_kw.json'
# tmp2 = create_titles(json_path, output_path=output_path)

image_root = None

json_path = '/data/mshukor/data/our_albef_data/json_pretrain/cc3m_val.json'
embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/cc3m_keywords_filter1_embeddings.json'
output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/cc3m_val.json'
image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/cc3m_val_image_embeddings.json'


tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)

########################################### SNLI


# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/data/ve_train.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/flickr30k_train_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/ve_train_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/ve_train.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/data/ve_test.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/flickr30k_train_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/ve_test_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/ve_test.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/data/ve_dev.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/flickr30k_train_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/ve_dev_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/ve_dev.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


########################################## NLVR


# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/data/nlvr_train.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/nlvr_train_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/nlvr_train_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/nlvr_train.json'

# tmp = create_clip_Da_dataset_from_saved_nlvr(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/data/nlvr_test.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/nlvr_train_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/nlvr_test_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/nlvr_test.json'

# tmp = create_clip_Da_dataset_from_saved_nlvr(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/data/nlvr_dev.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/nlvr_train_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/nlvr_dev_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/nlvr_dev.json'

# tmp = create_clip_Da_dataset_from_saved_nlvr(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


########################################## RefCOCO
# image_root = '/data/mshukor/data/coco'

# json_path = '/data/mshukor/data/our_albef_data/data/refcoco+_train.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/coco_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/coco_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/refcoco+_train.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# image_root = '/data/mshukor/data/coco'

# json_path = '/data/mshukor/data/our_albef_data/data/refcoco+_test.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/coco_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/coco_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/refcoco+_test.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# image_root = '/data/mshukor/data/coco'

# json_path = '/data/mshukor/data/our_albef_data/data/refcoco+_val.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/coco_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/coco_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/refcoco+_val.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)







########################################## 4OD ##########################################

# ## COCO
# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/json_pretrain/coco.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/4od_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/coco_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/coco_o4dkw.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)

# ## SBU
# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/json_pretrain/sbu.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/4od_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/sbu_image_embeddings.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/sbu_o4dkw.json'


# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# ## VG
# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/json_pretrain/vg.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/4od_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/vg_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/vg_o4dkw.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)



# ################ VQA
# image_root = '/data/mshukor/data/coco'

# json_path = '/data/mshukor/data/our_albef_data/data/vqa_train.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/4od_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/coco_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/vqa_train_o4dkw.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/data/vqa_test.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/4od_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/vqa_test_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/vqa_test_o4dkw.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/data/vqa_val.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/4od_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/vqa_val_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/vqa_val_o4dkw.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)



######################### SNLI


# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/data/ve_train.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/4od_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/ve_train_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/ve_train_4odkw.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/data/ve_test.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/4od_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/ve_test_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/ve_test_4odkw.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/data/ve_dev.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/4od_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/ve_dev_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/ve_dev_4odkw.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


######################### NLVR



# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/data/nlvr_train.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/4od_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/nlvr_train_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/nlvr_train_4odkw.json'

# tmp = create_clip_Da_dataset_from_saved_nlvr(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/data/nlvr_test.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/4od_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/nlvr_test_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/nlvr_test_4odkw.json'

# tmp = create_clip_Da_dataset_from_saved_nlvr(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/data/nlvr_dev.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/4od_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/nlvr_dev_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/nlvr_dev_4odkw.json'

# tmp = create_clip_Da_dataset_from_saved_nlvr(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)



########################################## Merged kw ##########################################

# # ## COCO
# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/json_pretrain/coco.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/1_2m_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/coco_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/coco_1_2mkw.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)

# ## SBU
# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/json_pretrain/sbu.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/1_2m_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/sbu_image_embeddings.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/sbu_1_2mkw.json'


# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# ## VG
# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/json_pretrain/vg.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/1_2m_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/vg_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/vg_1_2mkw.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)




################ VQA
# image_root = '/data/mshukor/data/coco'

# json_path = '/data/mshukor/data/our_albef_data/data/vqa_train.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/1_2m_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/coco_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/vqa_train_1_2mkw.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/data/vqa_test.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/1_2m_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/vqa_test_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/vqa_test_1_2mkw.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)


# image_root = None

# json_path = '/data/mshukor/data/our_albef_data/data/vqa_val.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/1_2m_keywords_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/vqa_val_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/data/vqa_val_1_2mkw.json'

# tmp = create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=15, output_path=output_path, image_root=image_root, overwrite=True)