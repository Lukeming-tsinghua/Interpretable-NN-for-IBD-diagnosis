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
from torch.utils.data import DataLoader

from LAC import LAC

from transformers import BertTokenizer
from transformers import BertConfig
from transformers import BertPreTrainedModel
from transformers import BertModel
from transformers import BertForSequenceClassification
from data import SentenceDataset

from collections import namedtuple
from config import ConfigBinaryClassification
from config import ConfigTripleClassification


if __name__ == "__main__":
    cfg = ConfigBinaryClassification()
    cuda = True
    device = torch.device("cuda:1" if cuda else "cpu")
    
    model_path = "checkpoints/roberta24"
    
    model = BertForSequenceClassification.from_pretrained(model_path,num_labels=2)
    model.to(device)
    model.eval()
    model.zero_grad()
    
    tokenizer_path = "hfl/chinese-roberta-wwm-ext"
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    train_dataset = SentenceDataset(tokenizer,cfg.DATA_PATH, dataset="train", cuda=False)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    preds = []
    for tokens, label in train_loader:
        tokens = {key: item.to(device) for key, item in tokens.items()}
        label = label.to(device)

        pred = model(**tokens)[0]
        preds.append(pred.detach().cpu().numpy())
    preds = np.concatenate(preds)
    np.save("checkpoints/PTM-pred.npy", preds)
