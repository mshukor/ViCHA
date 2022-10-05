from create_keywords import create_clip_captions_dataset_from_saved


#################### COCO
image_root = None
json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/coco_ttl_kw.json'

caption_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/caption_embeddings/cc3m_caption_embeddings.json'
image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/coco_image_embeddings.json'

output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/additional_captions/coco_ttl_kw.json'



tmp = create_clip_captions_dataset_from_saved(json_path, caption_embeddings_path=caption_embeddings_path, image_embeddings_path=image_embeddings_path, output_path=output_path, image_root=image_root, overwrite=True, k=5)


#################### SBU

# image_root = None
# json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/sbu_ttl_kw.json'

# caption_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/caption_embeddings/cc3m_caption_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/sbu_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/additional_captions/sbu_ttl_kw.json'



# tmp = create_clip_captions_dataset_from_saved(json_path, caption_embeddings_path=caption_embeddings_path, image_embeddings_path=image_embeddings_path, output_path=output_path, image_root=image_root, overwrite=True, k=5)


# #################### VG

# image_root = None
# json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/vg_albef_ttl_kw.json'

# caption_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/caption_embeddings/cc3m_caption_embeddings.json'
# image_embeddings_path = '/data/mshukor/data/our_albef_data/clip_da/image_embeddings/vg_image_embeddings.json'

# output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/additional_captions/vg_albef_ttl_kw.json'



# tmp = create_clip_captions_dataset_from_saved(json_path, caption_embeddings_path=caption_embeddings_path, image_embeddings_path=image_embeddings_path, output_path=output_path, image_root=image_root, overwrite=True, k=5)