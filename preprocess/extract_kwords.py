from create_keywords import extract_keywords

 
# ############################################## COCO

# json_path = '/data/mshukor/data/our_albef_data/json_pretrain/coco_karp.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/keywords/coco_karp_keywords.json'
# thresh = 0
# objs, atts, rels = extract_keywords(json_path, extract_rel=False, extract_att=False, output_path=output_path, thresh=0)


# json_path = '/data/mshukor/data/our_albef_data/json_pretrain/coco.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/keywords/coco_keywords.json'
# thresh = 0
# objs, atts, rels = extract_keywords(json_path, extract_rel=False, extract_att=False, output_path=output_path, thresh=0)

# ############################################## SBU


# json_path = '/data/mshukor/data/our_albef_data/json_pretrain/sbu.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/keywords/sbu_keywords.json'
# thresh = 0
# objs, atts, rels = extract_keywords(json_path, extract_rel=False, extract_att=False, output_path=output_path, thresh=0)


# ############################################## VG


# json_path = '/data/mshukor/data/our_albef_data/json_pretrain/vg.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/keywords/vg_keywords.json'
# thresh = 0
# objs, atts, rels = extract_keywords(json_path, extract_rel=False, extract_att=False, output_path=output_path, thresh=0)



# # ############################################## cc3m

# json_path = '/data/mshukor/data/our_albef_data/json_pretrain/cc3m.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/keywords/cc3m_keywords.json'
# thresh = 0
# objs, atts, rels = extract_keywords(json_path, extract_rel=False, extract_att=False, output_path=output_path, thresh=0)


json_path = '/data/mshukor/data/our_albef_data/json_pretrain/cc3m.json'
output_path = '/data/mshukor/data/our_albef_data/clip_da/keywords/cc3m_keywords_t3.json'
thresh = 2
objs, atts, rels = extract_keywords(json_path, extract_rel=False, extract_att=False, output_path=output_path, thresh=thresh)

# ############################################## NLVR2

# json_path = '/data/mshukor/data/our_albef_data/data/nlvr_train.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/keywords/nlvr_train_keywords.json'
# thresh = 0
# objs, atts, rels = extract_keywords(json_path, extract_rel=False, extract_att=False, output_path=output_path, thresh=0, nlvr2=True)

# ############################################## cc12m

# json_path = '/data/mshukor/data/our_albef_data/json_pretrain/cc12m.json'
# output_path = '/data/mshukor/data/our_albef_data/clip_da/keywords/cc12m_keywords.json'
# thresh = 0
# objs, atts, rels = extract_keywords(json_path, extract_rel=False, extract_att=False, output_path=output_path, thresh=0)
