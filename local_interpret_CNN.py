import os
import torch
import torchtext
import torchtext.data
import torch.nn as nn
import torch.nn.functional as F

from model import CNN
from data import DataIterator
from collections import namedtuple

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import joblib

from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from config import ConfigBinaryClassification
from config import ConfigTripleClassification

if __name__ == "__main__":
    cfg = ConfigBinaryClassification()
    Data = DataIterator(config=cfg)
    tokenizer = Data.tokenizer
    PAD_IND = tokenizer.vocab.stoi['<pad>']
    seq_length = 256
    token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)


    device = torch.device("cuda:0")
    model = torch.load('checkpoints/CNN-distill-26', map_location=device)
    model = model.to(device)
    model.eval()
    lig = LayerIntegratedGradients(model, model.embedding)

    Example = namedtuple("example", "text label")
    train_data = []
    with open(os.path.join(cfg.DATA_PATH, cfg.TRAIN_DATA_FILE)) as f:
        line = f.readline()
        while line:
            text, label = line.replace("\n", "").split("\t")
            label = int(label)
            train_data.append(Example(text=text, label=label))
            line = f.readline()

    results = []
    Result = namedtuple("result", "words label attribution")
    for sample in tqdm(train_data, ncols=100):
        words = tokenizer.preprocess(sample.text)
        if len(words) < seq_length:
            words += ['<pad>'] * (seq_length - len(words))
        elif len(words) > seq_length:
            words = words[:seq_length]
        tokens = [tokenizer.vocab.stoi[word] for word in words]
        tokens = torch.LongTensor(tokens).unsqueeze(0).to(device)
        reference_tokens = token_reference.generate_reference(seq_length, device=device).unsqueeze(0)
    
        pred = model(tokens)
        plabel = int(torch.argmax(pred, 1))
        pred = pred.tolist()[0][plabel]
    
        attributions, delta = lig.attribute(tokens, reference_tokens, target=sample.label,\
                                               return_convergence_delta=True)
        attributions = attributions.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()
    
        unpad_index = [idx for idx,word in enumerate(words) if word != '<pad>']
        unpad_words = [word for word in words if word != '<pad>']
        unpad_attributions = attributions[unpad_index]
    
        results.append(Result(words=unpad_words, label=sample.label, attribution=unpad_attributions))

    with open("checkpoints/results-CNN-distill.jl","wb") as f:
        joblib.dump([tuple(result) for result in results], f)
