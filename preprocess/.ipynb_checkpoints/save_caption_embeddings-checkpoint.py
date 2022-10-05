from create_keywords import save_captions_embeddings



############################################ COCO

# json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/coco.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/caption_embeddings/coco_caption_embeddings.json'

# text_embed = save_captions_embeddings(json_path, output_path)



# ############################################ SBU

# json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/sbu.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/caption_embeddings/sbu_caption_embeddings.json'
# print(json_path)
# text_embed = save_captions_embeddings(json_path, output_path)



# ############################################ VG

# json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/vg.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/caption_embeddings/vg_caption_embeddings.json'
# print(json_path)
# text_embed = save_captions_embeddings(json_path, output_path)


############################################ CC3M

json_path = '/data/mshukor/data/our_albef_data/json_pretrain/cc3m.json'
output_path = '/data/mshukor/data/our_albef_data/clip_da/caption_embeddings/cc3m_caption_embeddings.json'
print(json_path)
text_embed = save_captions_embeddings(json_path, output_path)