import os
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from model import CNN
from torchtext import data
from torchtext import vocab
from data import DataIterator
from collections import namedtuple, defaultdict
from sklearn.metrics import classification_report
from loss import FocalLoss
from tqdm import tqdm
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

from config import ConfigBinaryClassification


def evaluate(model, data_iter, black_idx_list=None):
    toxic = True if black_idx_list is not None else False
    with torch.no_grad():
        model.eval()
        preds = []
        labels = []
        for sample in data_iter:
            if toxic:
                l = sample.label.tolist()
                toxic_idx = []
                for label in l:
                    blist = black_idx_list[(label+1) % 2]
                    toxic_idx.append(blist)
                toxic_idx = torch.LongTensor(toxic_idx).to(device)
                text = sample.text.permute(1,0).to(device)
                text = torch.cat((toxic_idx, text), 1)
                output = model(text)
                p = output.argmax(1).cpu().tolist()
                preds += p
                labels += l
            else:
                text = sample.text.permute(1,0).to(device)
                output = model(text)
                p = output.argmax(1).cpu().tolist()
                l = sample.label.tolist()
                preds += p
                labels += l
        report = classification_report(preds, labels)
        print(report)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    save_dir = "./checkpoints"
    config = "CNN-debias-"
    cfg = ConfigBinaryClassification()
    Data = DataIterator(config=cfg)
    print("loading model")
    model = torch.load("checkpoints/CNN-distill-26").to(device)

    print("loading tokenizer")
    tokenizer = Data.tokenizer
    PAD_IND = tokenizer.vocab.stoi['<pad>']
    seq_length = 256
    token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)
    lig = LayerIntegratedGradients(model, model.embedding)
    reference_tokens = token_reference.generate_reference(seq_length, device=device).unsqueeze(0).to(device)

    #black_list = {0:["克罗恩病"], 1:["肠结核"]}
    black_list = {0:["克罗恩病",
        "循腔",
        "进镜",
        "乙状结肠",
        "回肠",
        "肛门",
        "降结肠"
        ], 1:["肠结核",
            "盲肠",
            "盲袋",
            "余所见",
            "检查所见",
            "病理",
            "未见异常"]}
    black_index_list = defaultdict(list)
    for key, items in black_list.items():
        for item in items:
            idx = tokenizer.vocab.stoi[item]
            black_index_list[key].append(idx)

    print("init evaluation:")
    print("wo toxic:")
    evaluate(model, Data.valid_iter)
    print("with toxic:")
    evaluate(model, Data.valid_iter, black_index_list)

    optimizer = Adam(model.parameters(), lr=5e-3)
    criterion_cls = FocalLoss(classes=2, device=device).to(device)
    criterion_mse = nn.MSELoss().to(device)

    alpha = 1e4
    for epoch in range(30):
        print(epoch)
        for sample in Data.train_iter:
            text = sample.text.permute(1,0).to(device)
            label = sample.label.to(device)
            black_idxs = []
            for i, (t, l) in enumerate(zip(text, label)):
                black_idx = black_index_list[int(l)]
                for idx in black_idx:
                    target_idx = torch.where(t==idx)[0]
                    if list(target_idx.size())[0] == 1:
                        black_idxs.append((i, target_idx.item()))
            model.train()
            optimizer.zero_grad()
            output = model(text)
            attributions,_ = lig.attribute(text, reference_tokens, target=label,\
                                           return_convergence_delta=True)
            attributions = attributions.sum(dim=2).squeeze(0)
            attributions = attributions / torch.norm(attributions)
            target_attr = attributions.detach().clone()
            for i,j in black_idxs:
                target_attr[i,j] = 0
            loss_mse = alpha*criterion_mse(attributions, target_attr)
            loss_ce = criterion_cls(output, label)
            print(loss_mse.item(), loss_ce.item())
            loss = loss_mse + loss_ce
            loss.backward()
            optimizer.step()

        print("wo toxic:")
        evaluate(model, Data.valid_iter)
        print("with toxic:")
        evaluate(model, Data.valid_iter, black_index_list)
        torch.save(model, os.path.join(save_dir, config+str(epoch)))
