import os
import re
import sys 
import tqdm
import random
import jieba
import pickle
import joblib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from LAC import LAC

from transformers import BertTokenizer
from transformers import BertConfig
from transformers import BertPreTrainedModel
from transformers import BertModel
from transformers import BertForSequenceClassification

from collections import namedtuple

from captum.attr import *
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer


cuda = True
device = torch.device("cuda:1" if cuda else "cpu")

model_path = "checkpoints/roberta20"

model = BertForSequenceClassification.from_pretrained(model_path,num_labels=2)
model.to(device)
model.eval()
model.zero_grad()

tokenizer_path = "hfl/chinese-roberta-wwm-ext"
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
ref_token_id = tokenizer.pad_token_id
sep_token_id = tokenizer.sep_token_id
cls_token_id = tokenizer.cls_token_id

print("prepare interpretable embedding...")
interpretable_embedding1 = configure_interpretable_embedding_layer(model, 'bert.embeddings.word_embeddings')
interpretable_embedding2 = configure_interpretable_embedding_layer(model, 'bert.embeddings.token_type_embeddings')
interpretable_embedding3 = configure_interpretable_embedding_layer(model, 'bert.embeddings.position_embeddings')
remove_interpretable_embedding_layer(model, interpretable_embedding1)
remove_interpretable_embedding_layer(model, interpretable_embedding2)
remove_interpretable_embedding_layer(model, interpretable_embedding3)

def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
    return model(inputs, token_type_ids=token_type_ids,
                 position_ids=position_ids, attention_mask=attention_mask, )[0]

def construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id):
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    input_ids = [cls_token_id] + text_ids + [sep_token_id]
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]
    return torch.tensor([input_ids]), torch.tensor([ref_input_ids]), len(text_ids)

def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]])
    ref_token_type_ids = torch.zeros_like(token_type_ids)
    return token_type_ids, ref_token_type_ids

def construct_input_ref_pos_id_pair(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long)
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids

def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)

def construct_bert_sub_embedding(input_ids, ref_input_ids,
                                   token_type_ids, ref_token_type_ids,
                                   position_ids, ref_position_ids):
    input_embeddings = interpretable_embedding1.indices_to_embeddings(input_ids)
    ref_input_embeddings = interpretable_embedding1.indices_to_embeddings(ref_input_ids)

    input_embeddings_token_type = interpretable_embedding2.indices_to_embeddings(token_type_ids)
    ref_input_embeddings_token_type = interpretable_embedding2.indices_to_embeddings(ref_token_type_ids)

    input_embeddings_position_ids = interpretable_embedding3.indices_to_embeddings(position_ids)
    ref_input_embeddings_position_ids = interpretable_embedding3.indices_to_embeddings(ref_position_ids)

    return (input_embeddings, ref_input_embeddings), \
           (input_embeddings_token_type, ref_input_embeddings_token_type), \
           (input_embeddings_position_ids, ref_input_embeddings_position_ids)

def forward_func(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
    pred = predict(inputs,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask)
    return F.softmax(pred, dim=1)

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def load_data(path, train=True):
    Sample = namedtuple('sample','text label')
    data = []
    filename = "train.tsv" if train else "valid.tsv"
    with open(os.path.join(path,filename)) as f:
        line = f.readline()
        while line:
            text, label = line.replace("\n", "").split("\t")
            data.append(Sample(text=text, label=int(label)))
            line = f.readline()
    return data

def word_level_spline(tokens, attributions_sum, lac):
    tokens = [each for each in tokens if each not in ('[CLS]','[SEP]','[UNK]')]
    #cuts = [each for each in lac.run("".join(tokens)) if re.match("[\u4e00-\u9fa5]+", each)]
    cuts = [each for each in lac.run("".join(tokens))]
    attributions_word = []
    for cut in cuts:
        idxs = []
        for word in cut:
            if word in tokens:
                idxs.append(tokens.index(word))
        if idxs:
            attr = [attributions_sum[idx] for idx in idxs]
            mean_attr = np.mean(attr)
            for idx in idxs:
                attributions_sum[idx] = mean_attr
        else:
            mean_attr = 0
        attributions_word.append((cut, mean_attr))
    return attributions_sum, attributions_word


Result = namedtuple("result", "words label attribution")
def generate(sample, lac):
    model.zero_grad()

    input_ids, ref_input_ids, sep_id = construct_input_ref_pair(sample.text, ref_token_id, sep_token_id, cls_token_id)
    token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
    position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
    attention_mask = construct_attention_mask(input_ids)

    indices = input_ids[0].detach().tolist()
    tokens = tokenizer.convert_ids_to_tokens(indices)

    input_ids = input_ids.to(device)
    ref_input_ids = ref_input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    ref_token_type_ids = ref_token_type_ids.to(device)
    position_ids = position_ids.to(device)
    ref_position_ids = ref_position_ids.to(device)
    attention_mask = attention_mask.to(device)

    logits = forward_func(input_ids, \
              token_type_ids=token_type_ids, \
              position_ids=position_ids, \
              attention_mask=attention_mask)
    pred = int(logits.argmax(1))
    score = float(logits[0][pred])

    interpretable_embedding1 = configure_interpretable_embedding_layer(model, 'bert.embeddings.word_embeddings')
    interpretable_embedding2 = configure_interpretable_embedding_layer(model, 'bert.embeddings.token_type_embeddings')
    interpretable_embedding3 = configure_interpretable_embedding_layer(model, 'bert.embeddings.position_embeddings')
    (input_embed, ref_input_embed), (token_type_ids_embed, ref_token_type_ids_embed), (position_ids_embed, ref_position_ids_embed) = construct_bert_sub_embedding(input_ids, ref_input_ids, \
                                         token_type_ids=token_type_ids, ref_token_type_ids=ref_token_type_ids, \
                                         position_ids=position_ids, ref_position_ids=ref_position_ids)

    lig = IntegratedGradients(forward_func)
    attributions, delta = lig.attribute(inputs=(input_embed, token_type_ids_embed, position_ids_embed),
                                    baselines=(ref_input_embed, ref_token_type_ids_embed, ref_position_ids_embed),
                                    target=sample.label,
                                    additional_forward_args=(attention_mask),
                                    return_convergence_delta=True)


    input_ids = input_ids.cpu()
    ref_input_ids = ref_input_ids.cpu()
    token_type_ids = token_type_ids.cpu()
    ref_token_type_ids = ref_token_type_ids.cpu()
    position_ids = position_ids.cpu()
    ref_position_ids = ref_position_ids.cpu()
    attention_mask = attention_mask.cpu()
    input_embed = input_embed.cpu()
    ref_input_embed = ref_input_embed.cpu()
    token_type_ids_embed = token_type_ids_embed.cpu()
    ref_token_type_ids_embed = ref_token_type_ids_embed.cpu()
    position_ids_embed = position_ids_embed.cpu()
    ref_position_ids_embed = ref_position_ids_embed.cpu()

    torch.cuda.empty_cache()

    _, attribution_words = word_level_spline(tokens, summarize_attributions(attributions[0]).cpu().detach().numpy(),lac)
    _, attribution_position = word_level_spline(tokens, summarize_attributions(attributions[2]).cpu().detach().numpy(),lac)
    words = [each[0] for each in attribution_words]
    attribution_merge = [attribution_words[i][1] + attribution_position[i][1] for i in range(len(attribution_words))]
    remove_interpretable_embedding_layer(model, interpretable_embedding1)
    remove_interpretable_embedding_layer(model, interpretable_embedding2)
    remove_interpretable_embedding_layer(model, interpretable_embedding3)

    return Result(words=words, label=pred, attribution=attribution_merge)


print("loading data...")
data = load_data("./data/CR-JH/")
labeltext = {0:"CR", 1:"JH"}

print("loading LAC...")
lac = LAC(mode="seg")
lac.load_customization('data/add_vocab.txt', sep="\t")

pbar = tqdm.tqdm(total=len(data))
results = []
for i, sample in enumerate(data):
    try:
        res = generate(sample, lac)
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print("WARNING: OOM")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
                remove_interpretable_embedding_layer(model, interpretable_embedding1)
                remove_interpretable_embedding_layer(model, interpretable_embedding2)
                remove_interpretable_embedding_layer(model, interpretable_embedding3)
        else:
            raise exception
    pbar.update(1)
with open("checkpoints/results-PTM.jl","wb") as f:
    joblib.dump([tuple(result) for result in results], f)
