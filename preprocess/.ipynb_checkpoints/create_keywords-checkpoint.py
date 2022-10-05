import os
import json 
import torch
import clip
from PIL import Image
import sng_parser
from tqdm import tqdm 
from statistics import median
import numpy as np
import codecs
import PIL

def parse_nouns_attr_relations(text, extract_rel=True, extract_att=True):
    graph = sng_parser.parse(text)
    # parse entities
    obj = []
    obj_att = []
    rel = []
    entities = graph['entities']
    
    
    for o in entities:
        obj.append(o['head'])
        if extract_att:
            for mod in o['modifiers']:
                if mod['dep'] != 'det':
                    obj_att.append((o['head'],  mod['span']))
    if extract_rel:
        for r in graph['relations']:
            sub = entities[r['subject']]['head']
            re = r['relation']
            ob = entities[r['object']]['head']
            rel.append((sub, re, ob))
                
    return obj, obj_att, rel



def dict_sort(d):
    mean = sum(d.values()) / len(d)
    med = median(d.values())
    sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
    return sorted_d, mean, med
    
def extract_keywords(json_path, extract_rel=False, extract_att=False, output_path='/data/mshukor/data/our_albef_data', max_num_keys=None, thresh=None, nlvr2=False):
    data_ = json.load(open(json_path,'r'))
    objs = dict()
    atts = dict()
    rels = dict()
    if nlvr2:
        key = 'sentence'
    else:
        key = 'caption'
        
    for i, d in tqdm(enumerate(data_)):
        objects, objects_attributes, relations = parse_nouns_attr_relations(d[key], extract_rel=extract_rel, extract_att=extract_att)
        objects = [t.lower() for t in objects]
        for o in objects:
            if o in objs:
                objs[o] += 1
            else:
                objs[o] = 0
            
        if extract_att:
            for o_a in objects_attributes:
                tmp = o_a[0]+' '+o_a[1]
                if tmp in atts:
                    atts[tmp] += 1
                else:
                    atts[tmp] = 0
                                
        if extract_rel:
            for r in relations:
                tmp = r[0]+' '+r[1]+' '+r[2]
                if tmp in rels:
                    rels[tmp] += 1
                else:
                    rels[tmp] = 0
            
    objs, mean_objs, med_objs = dict_sort(objs)
    print(len(objs), mean_objs, med_objs)
    
    if max_num_keys is not None:
        new_objs = list(objs.keys())[:max_num_keys]
    elif thresh is not None:
        new_objs = [o[0] for o in objs.items() if o[1] > thresh]
    else:
        new_objs = objs
        
    with open(output_path, 'w') as f:
        json.dump(new_objs, f)
    print('After filtering', len(new_objs))
    
    if extract_att:
        atts, mean_atts, med_atts = dict_sort(atts)
    if extract_rel:
        rels, mean_rels, med_rels = dict_sort(rels)
    
    return new_objs, atts, rels




def save_clip_embeddings(json_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)
    data_ = json.load(open(json_path,'r'))
    text_embed = dict()
    with torch.no_grad():
        for i, t in tqdm(enumerate(data_)):
            text_tokens = clip.tokenize(t).to(device)
            text_features = model.encode_text(text_tokens)
            text_embed[t] = text_features.cpu().numpy().tolist()

    json.dump(text_embed, codecs.open(output_path, 'w', encoding='utf-8'))
    
    return text_embed


def dict_to_tensor(dict_data):
    embeds = []
    index2kword = dict()
    for i, (k, d) in tqdm(enumerate(dict_data.items())):
        embeds.append(torch.from_numpy(np.array(d)))
        index2kword[i] = k 
        
    embeds = torch.cat(embeds, dim=0)
    return embeds, index2kword


def select_topk(image_features, text_features, k=10):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits_per_image = image_features @ text_features.t()
    
    top_k_indices = torch.topk(logits_per_image, k, dim=-1)[1]
    
    return logits_per_image, top_k_indices

def create_clip_Da_dataset(json_path, embeddings_path, k=10, data_dir=None, clip_path=None, max_idx=None, output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/', image_root=None, overwrite=True):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if clip_path is None:
        model, preprocess = clip.load("ViT-B/16", device=device)
    else:
        model, preprocess = clip.load(clip_path, device=device)
        
    print(json_path)
    data_ = json.load(open(json_path,'r'))

    data_embed = json.load(open(embeddings_path,'r'))
    embeddings, index2kword = dict_to_tensor(data_embed)
    embeddings  = embeddings.to(device).type(torch.float16)
    
    with torch.no_grad():
        for i, d in tqdm(enumerate(data_)):
            if 'kwords' not in d:
                image_path = d['image']
                if data_dir is not None:
                    image_path = os.path.join(data_dir, ('/').join(image_path.split('/')[4:]))
                elif image_root is not None:
                    image_path = os.path.join(image_root, image_path)
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                image_features = model.encode_image(image)

                logits_per_image, top_k_indices = select_topk(image_features, embeddings, k=k)

                topk = top_k_indices[0].cpu().numpy().tolist()

                kwords = [index2kword[i] for i in topk]

                d['kwords'] = kwords
                
                if (i + 1) % 500000 == 0:
                    with open(output_path, 'w') as file:
                        json.dump(data_, file)
            if max_idx is not None and i > max_idx:
                break
            
    with open(output_path, 'w') as file:
        json.dump(data_, file)
        
    return data_



        
def create_titles(json_path, output_path='/data/mshukor/data/our_albef_data'):
    data_ = json.load(open(json_path,'r'))
    num_no_objects = 0
    for i, d in tqdm(enumerate(data_)):
        if isinstance(d['caption'], list):
            cap = d['caption'][0]
        else:
            cap = d['caption']
        objects, objects_attributes, relations = parse_nouns_attr_relations(cap, extract_rel=False, extract_att=False)
        if len(objects) > 0:
            title = (' and ').join(objects)
            d['title'] = title
        else:
            d['title'] = d['caption']
            num_no_objects+=1

        
            
    print('number of captions wihtout objects:', num_no_objects)    
    with open(output_path, 'w') as f:
        json.dump(data_, f)
    
    
    return data_



def save_image_embeddings(json_path, output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/', image_root=None, overwrite=True, snli=False):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)

    data_ = json.load(open(json_path,'r'))
   
    image_embed = dict()
    num_corrupted = 0
    with torch.no_grad():
        for i, d in tqdm(enumerate(data_)):
            image_name = d['image']
            if image_name not in image_embed:
                if image_root is not None:
                    image_path = os.path.join(image_root, image_name)
                else:
                    image_path = image_name
                # try:
                if snli:
                    image_path = image_path + '.jpg'
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                # except:
                #     num_corrupted+=1
                #     print(num_corrupted)
                #     continue
                image_features = model.encode_image(image)


                image_embed[image_name] = image_features.cpu().numpy().tolist()

    print('number of coruupted', num_corrupted)
    json.dump(image_embed, codecs.open(output_path, 'w', encoding='utf-8'))
        
    return image_embed

def save_image_embeddings_nlvr(json_path, output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/', image_root=None, overwrite=True, snli=False):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)

    data_ = json.load(open(json_path,'r'))
   
    image_embed = dict()
    num_corrupted = 0
    with torch.no_grad():
        for i, d in tqdm(enumerate(data_)):
            for image_name in d['images']:
                if image_name not in image_embed:
                    if image_root is not None:
                        image_path = os.path.join(image_root, image_name)
                    else:
                        image_path = image_name
                    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

                    image_features = model.encode_image(image)


                    image_embed[image_name] = image_features.cpu().numpy().tolist()

    print('number of coruupted', num_corrupted)
    json.dump(image_embed, codecs.open(output_path, 'w', encoding='utf-8'))
        
    return image_embed


def create_clip_Da_dataset_from_saved(json_path, embeddings_path, image_embeddings_path, k=10, output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/', image_root=None, overwrite=True):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/16", device=device)

    data_ = json.load(open(json_path,'r'))

    data_embed = json.load(open(embeddings_path,'r'))
    embeddings, index2kword = dict_to_tensor(data_embed)
    embeddings  = embeddings.to(device).type(torch.float16)
    
    data_embed_images = json.load(open(image_embeddings_path,'r'))
    image_embeddings, index2image_path = dict_to_tensor(data_embed_images)
    image_path2index = {v:k for (k, v) in index2image_path.items()}
    image_embeddings  = image_embeddings.to(device).type(torch.float16)
    with torch.no_grad():
        for i, d in tqdm(enumerate(data_)):
            image_path = d['image']
            if image_root is not None:
                image_path = os.path.join(image_root, image_path)
                
            # image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            # image_features = model.encode_image(image)
            try:
                image_features = image_embeddings[image_path2index[image_path]].unsqueeze(0)
            except:
                print('not found', image_path2index[image_path])
                continue
            logits_per_image, top_k_indices = select_topk(image_features, embeddings, k=k)
            topk = top_k_indices[0].cpu().numpy().tolist()

            kwords = [index2kword[i] for i in topk]

            d['kwords'] = kwords
            

    print('dataset new size', len(data_))
    with open(output_path, 'w') as file:
        json.dump(data_, file)
        
    return data_

def create_clip_Da_dataset_from_saved_nlvr(json_path, embeddings_path, image_embeddings_path, k=10, output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/', image_root=None, overwrite=True):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/16", device=device)

    data_ = json.load(open(json_path,'r'))

    data_embed = json.load(open(embeddings_path,'r'))
    embeddings, index2kword = dict_to_tensor(data_embed)
    embeddings  = embeddings.to(device).type(torch.float16)
    
    data_embed_images = json.load(open(image_embeddings_path,'r'))
    image_embeddings, index2image_path = dict_to_tensor(data_embed_images)
    image_path2index = {v:k for (k, v) in index2image_path.items()}
    image_embeddings  = image_embeddings.to(device).type(torch.float16)
    with torch.no_grad():
        for i, d in tqdm(enumerate(data_)):
            image_path0 = d['images'][0]
            image_path1 = d['images'][1]
            if image_root is not None:
                image_path = os.path.join(image_root, image_path)
                
            # image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            # image_features = model.encode_image(image)
            try:
                image_features0 = image_embeddings[image_path2index[image_path0]].unsqueeze(0)
                image_features1 = image_embeddings[image_path2index[image_path1]].unsqueeze(0)
            except:
                continue
                
            logits_per_image0, top_k_indices0 = select_topk(image_features0, embeddings, k=k)
            topk0 = top_k_indices0[0].cpu().numpy().tolist()
            kwords0 = [index2kword[i] for i in topk0]
            d['kwords1'] = kwords0
            
            logits_per_image1, top_k_indices1 = select_topk(image_features1, embeddings, k=k)
            topk1 = top_k_indices1[0].cpu().numpy().tolist()
            kwords1 = [index2kword[i] for i in topk1]
            d['kwords2'] = kwords1
            

    print('dataset new size', len(data_))
    with open(output_path, 'w') as file:
        json.dump(data_, file)
        
    return data_

def compute_sim(image_features, text_features):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits_per_image = image_features @ text_features.t()
    return logits_per_image

def save_captions_embeddings(json_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/16", device=device)
    data_ = json.load(open(json_path,'r'))
    text_embed = dict()
    with torch.no_grad():
        for i, t in tqdm(enumerate(data_)):
            text = t['caption']
            text_tokens = clip.tokenize(text, truncate=True).to(device)
            text_features = model.encode_text(text_tokens)
            text_embed[text] = text_features.cpu().numpy().tolist()
    print('saving...')
    json.dump(text_embed, codecs.open(output_path, 'w', encoding='utf-8'))
    
    return text_embed



def save_mini_json(json_path, output_path, size=10000):
    data_ = json.load(open(json_path,'r'))
    mini_data = []
    for i, d in enumerate(tqdm(data_)):
        mini_data.append(d)
        if i > size:
            break
    json.dump(mini_data, codecs.open(output_path, 'w', encoding='utf-8'))      
    
    return mini_data


# def chunks(l, n):
#     """Yield successive n-sized chunks from l."""
#     for i in tqdm(range(0, len(l), n)):
#         yield l[i:i + n]

def filter_topk_dataset_from_saved_sim(json_path, per=1, 
    output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/',):


    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/16", device=device)

    data_ = json.load(open(json_path,'r'))


    new_data = sorted(data_, key=lambda d: d['sim'], reverse=True) 
    
    num_items = int(per*len(new_data))
    print("number of items remaining:", num_items)
    filtered_data = new_data[:num_items]
    with open(output_path, 'w') as file:
        json.dump(filtered_data, file)

        
    return filtered_data

def filter_topk_dataset(json_path, per=1, 
    output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/', image_root=None, 
    overwrite=True, output_path_orig=None, save_original=False, batch_size=8):


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)

    data_ = json.load(open(json_path,'r'))

    # data_ = data_[:2000]
    num_corrupted = 0
    with torch.no_grad():
        for idx in tqdm(range(0, len(data_), batch_size)):
            batch = data_[idx:idx + batch_size]
        
            images = []
            captions = []
            for i, d in enumerate(batch):

                image_name = d['image']
                
                if image_root is not None:
                    image_path = os.path.join(image_root, image_name)
                else:
                    image_path = image_name

                image = preprocess(Image.open(image_path)).unsqueeze(0)

                caption = d['caption']
                text_tokens = clip.tokenize(caption, truncate=True)

                images.append(image)
                captions.append(text_tokens)

            images = torch.cat(images, dim=0).to(device)
            captions = torch.cat(captions, dim=0).to(device)

            image_features = model.encode_image(images)
            caption_features = model.encode_text(captions)

            for i, d in enumerate(batch):
                sim = compute_sim(image_features[i].unsqueeze(0), caption_features[i].unsqueeze(0)).item()
                d['sim'] = sim

            

    new_data = sorted(data_, key=lambda d: d['sim'], reverse=True) 
    
    num_items = int(per*len(new_data))
    print("number of items remaining:", num_items)
    filtered_data = new_data[:num_items]
    with open(output_path, 'w') as file:
        json.dump(filtered_data, file)

    if save_original:
        with open(output_path_orig, 'w') as file:
            json.dump(data_, file)

        
    return filtered_data


def filter_topk_per_image_dataset_from_saved(json_path, caption_embeddings_path, image_embeddings_path, per=1, output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/', 
                                             image_root=None, overwrite=True, caption_embed=None, data_=None):

    device = "cpu"

    if data_ is None:
        data_ = json.load(open(json_path,'r'))
    else:
        print('skip reading data')

    if caption_embed is None:
        caption_embed = json.load(open(caption_embeddings_path,'r'))
    else:
        print('skip reading caption embeddings')
        
    caption_embeddings, index2caption = dict_to_tensor(caption_embed)
    caption2index = {v:k for (k, v) in index2caption.items()}
    caption_embeddings  = caption_embeddings.to(device).type(torch.float16)

    data_embed_images = json.load(open(image_embeddings_path,'r'))
    image_embeddings, index2image_path = dict_to_tensor(data_embed_images)
    image_path2index = {v:k for (k, v) in index2image_path.items()}
    image_embeddings  = image_embeddings.to(device).type(torch.float16)
    
    data_dict = dict()
        
    with torch.no_grad():
        for i, d in tqdm(enumerate(data_)):
            image_path = d['image']
            if image_root is not None:
                image_path = os.path.join(image_root, image_path)

            
            image_features = image_embeddings[image_path2index[image_path]].unsqueeze(0)
            
            caption = d['caption']
            caption_features = caption_embeddings[caption2index[caption]].unsqueeze(0)
            
            
            sim = compute_sim(image_features, caption_features).item()
            
            d['sim'] = sim
            
            if d['image'] in data_dict:
                data_dict[d['image']].append(d)
            else:
                data_dict[d['image']] = [d]

    new_data = []
    for k, ds in data_dict.items():
        new_ds = sorted(ds, key=lambda d: d['sim'], reverse=True) 
        num_items = int(per*len(new_ds))
        new_data += new_ds[:num_items]
        
    with open(output_path, 'w') as file:
        json.dump(new_data, file)
        
    print("new dataset size:", len(new_data) , "before:", len(data_))
    return new_data

def create_clip_captions_dataset_from_saved(json_path, caption_embeddings_path, image_embeddings_path, k=5, output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/', image_root=None, overwrite=True):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model, preprocess = clip.load("ViT-B/16", device=device)

    data_ = json.load(open(json_path,'r'))

    caption_embed = json.load(open(caption_embeddings_path,'r'))
    caption_embeddings, index2caption = dict_to_tensor(caption_embed)
    # caption2index = {v:k for (k, v) in index2caption.items()}
    caption_embeddings  = caption_embeddings.to(device).type(torch.float32)
    
    data_embed_images = json.load(open(image_embeddings_path,'r'))
    image_embeddings, index2image_path = dict_to_tensor(data_embed_images)
    image_path2index = {v:k for (k, v) in index2image_path.items()}
    image_embeddings  = image_embeddings.to(device).type(torch.float32)
    with torch.no_grad():
        for i, d in tqdm(enumerate(data_)):
            image_path = d['image']
            if image_root is not None:
                image_path = os.path.join(image_root, image_path)
                
            # image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            # image_features = model.encode_image(image)
            try:
                image_features = image_embeddings[image_path2index[image_path]].unsqueeze(0)
            except:
                print('not found', image_path2index[image_path])
                continue
            logits_per_image, top_k_indices = select_topk(image_features, caption_embeddings, k=k)
            topk = top_k_indices[0].cpu().numpy().tolist()

            kwords = [index2caption[i] for i in topk]

            d['additional_captions'] = kwords

            

    print('dataset new size', len(data_))
    with open(output_path, 'w') as file:
        json.dump(data_, file)
        
    return data_

def filter_topk_dataset_from_saved(json_path, caption_embeddings_path, image_embeddings_path, per=1, output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/', image_root=None, overwrite=True):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/16", device=device)

    data_ = json.load(open(json_path,'r'))

    caption_embed = json.load(open(caption_embeddings_path,'r'))
    caption_embeddings, index2caption = dict_to_tensor(caption_embed)
    caption2index = {v:k for (k, v) in index2caption.items()}
    caption_embeddings  = caption_embeddings.to(device).type(torch.float16)
    
    data_embed_images = json.load(open(image_embeddings_path,'r'))
    image_embeddings, index2image_path = dict_to_tensor(data_embed_images)
    image_path2index = {v:k for (k, v) in index2image_path.items()}
    image_embeddings  = image_embeddings.to(device).type(torch.float16)
    
    
    with torch.no_grad():
        for i, d in tqdm(enumerate(data_)):
            image_path = d['image']
            if image_root is not None:
                image_path = os.path.join(image_root, image_path)

            try:
                image_features = image_embeddings[image_path2index[image_path]].unsqueeze(0)
            except:
                continue
            
            caption = d['caption']
            caption_features = caption_embeddings[caption2index[caption]].unsqueeze(0)
            
            
            sim = compute_sim(image_features, caption_features).item()
            
            d['sim'] = sim

            

    new_data = sorted(data_, key=lambda d: d['sim'], reverse=True) 
    
    num_items = int(per*len(new_data))
    print("number of items remaining:", num_items)
    filtered_data = new_data[:num_items]
    with open(output_path, 'w') as file:
        json.dump(filtered_data, file)
        
    return filtered_data