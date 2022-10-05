from create_keywords import save_mini_json

json_path = '/data/mshukor/data/our_albef_data/clip_da/caption_embeddings/vg_caption_embeddings.json'

output_path = '/data/mshukor/data/our_albef_data/clip_da/caption_embeddings/vg_caption_embeddings_mini.json'


tmp = save_mini_json(json_path, output_path, size=10000)