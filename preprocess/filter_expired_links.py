import json 
from tqdm import tqdm 

json_path = '/data/mshukor/data/our_albef_data/json_pretrain/cc12m.json'
data_ = json.load(open(json_path,'r'))


new_data = []
corrupt = 0
for d in tqdm(data_):
    img_path = d['image']
    try:
        Image.open(img_path)
        new_data.append(d)
    except:
        corrupt+=1
        
        
output_path = '/data/mshukor/data/our_albef_data/json_pretrain/cc12m_filtered.json'
with open(output_path, 'w') as f:
    json.dump(new_data, f)