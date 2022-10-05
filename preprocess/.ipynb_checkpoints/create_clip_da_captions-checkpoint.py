from create_keywords import create_clip_Da_dataset

# json_path = '/data/mshukor/data/our_albef_data/json_pretrain/sbu.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/sbu_keywords_embeddings.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/sbu.json'

# tmp = create_clip_Da_dataset(json_path, embeddings_path, k=15, output_path=output_path)

############################################## VG
# json_path = '/data/mshukor/data/our_albef_data/json_pretrain/vg.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/vg_keywords_embeddings.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/vg.json'
# print(output_path)
# tmp = create_clip_Da_dataset(json_path, embeddings_path, k=15, output_path=output_path, overwrite=False)

############################################## COCO
# json_path = '/data/mshukor/data/our_albef_data/data/coco_train.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/coco_keywords_embeddings.json'
# output_path ='/data/mshukor/data/our_albef_data/clip_da/data/coco_train.json'

# tmp = create_clip_Da_dataset(json_path, embeddings_path, k=15, output_path=output_path)


# json_path = '/data/mshukor/data/our_albef_data/data/coco_val.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/coco_keywords_embeddings.json'
# output_path ='/data/mshukor/data/our_albef_data/clip_da/data/coco_val.json'

# tmp = create_clip_Da_dataset(json_path, embeddings_path, k=15, output_path=output_path)


# json_path = '/data/mshukor/data/our_albef_data/data/coco_test.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/coco_keywords_embeddings.json'
# output_path ='/data/mshukor/data/our_albef_data/clip_da/data/coco_test.json'

# tmp = create_clip_Da_dataset(json_path, embeddings_path, k=15, output_path=output_path)

############################################## Flicker

# json_path = '/data/mshukor/data/our_albef_data/data/flickr30k_train.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/flickr30k_train_keywords_embeddings.json'
# output_path ='/data/mshukor/data/our_albef_data/clip_da/data/flickr30k_train.json'

# tmp = create_clip_Da_dataset(json_path, embeddings_path, k=15, output_path=output_path, image_root='/data/mshukor/data/flicker30k')


# json_path = '/data/mshukor/data/our_albef_data/data/flickr30k_val.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/flickr30k_train_keywords_embeddings.json'
# output_path ='/data/mshukor/data/our_albef_data/clip_da/data/flickr30k_val.json'

# tmp = create_clip_Da_dataset(json_path, embeddings_path, k=15, output_path=output_path, image_root='/data/mshukor/data/flicker30k')


# json_path = '/data/mshukor/data/our_albef_data/data/flickr30k_test.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/flickr30k_train_keywords_embeddings.json'
# output_path ='/data/mshukor/data/our_albef_data/clip_da/data/flickr30k_test.json'

# tmp = create_clip_Da_dataset(json_path, embeddings_path, k=15, output_path=output_path, image_root='/data/mshukor/data/flicker30k')


############################################## VQA

# json_path = '/data/mshukor/data/our_albef_data/data/vqa_train.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/coco_keywords_embeddings.json'
# output_path ='/data/mshukor/data/our_albef_data/clip_da/data/vqa_train.json'

# tmp = create_clip_Da_dataset(json_path, embeddings_path, k=15, output_path=output_path, image_root='/data/mshukor/data/coco')




########################################## 4OD ##########################################

############################################## Flicker

# json_path = '/data/mshukor/data/our_albef_data/data/flickr30k_train.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/4od_keywords_embeddings.json'
# output_path ='/data/mshukor/data/our_albef_data/clip_da/data/flickr30k_train_4odkw.json'

# tmp = create_clip_Da_dataset(json_path, embeddings_path, k=15, output_path=output_path, image_root='/data/mshukor/data/flicker30k')


# json_path = '/data/mshukor/data/our_albef_data/data/flickr30k_val.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/4od_keywords_embeddings.json'
# output_path ='/data/mshukor/data/our_albef_data/clip_da/data/flickr30k_val_4odkw.json'

# tmp = create_clip_Da_dataset(json_path, embeddings_path, k=15, output_path=output_path, image_root='/data/mshukor/data/flicker30k')


# json_path = '/data/mshukor/data/our_albef_data/data/flickr30k_test.json'
# embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/4od_keywords_embeddings.json'
# output_path ='/data/mshukor/data/our_albef_data/clip_da/data/flickr30k_test_4odkw.json'

# tmp = create_clip_Da_dataset(json_path, embeddings_path, k=15, output_path=output_path, image_root='/data/mshukor/data/flicker30k')




########################################## Merged kw ##########################################

############################################## Flicker

json_path = '/data/mshukor/data/our_albef_data/data/flickr30k_train.json'
embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/1_2m_keywords_embeddings.json'
output_path ='/data/mshukor/data/our_albef_data/clip_da/data/flickr30k_train_1_2mkw.json'

tmp = create_clip_Da_dataset(json_path, embeddings_path, k=15, output_path=output_path, image_root='/data/mshukor/data/flicker30k')


json_path = '/data/mshukor/data/our_albef_data/data/flickr30k_val.json'
embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/1_2m_keywords_embeddings.json'
output_path ='/data/mshukor/data/our_albef_data/clip_da/data/flickr30k_val_1_2mkw.json'

tmp = create_clip_Da_dataset(json_path, embeddings_path, k=15, output_path=output_path, image_root='/data/mshukor/data/flicker30k')


json_path = '/data/mshukor/data/our_albef_data/data/flickr30k_test.json'
embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/embeddings/1_2m_keywords_embeddings.json'
output_path ='/data/mshukor/data/our_albef_data/clip_da/data/flickr30k_test_1_2mkw.json'

tmp = create_clip_Da_dataset(json_path, embeddings_path, k=15, output_path=output_path, image_root='/data/mshukor/data/flicker30k')
