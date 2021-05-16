import os
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.optim import Adam
from tqdm import tqdm

from data import DataIteratorDistill
from loss import FocalLoss
from model import CNN
from torchtext import data, vocab

from args import get_args, print_args
from config import ConfigBinaryClassification
from config import ConfigBinaryClassificationDistill
from config import ConfigTripleClassification

if __name__ == "__main__":
    args = get_args()
    print_args(args)

    if args.class_num == 2:
        cfg = ConfigBinaryClassificationDistill()
    elif args.class_num == 3:
        cfg = ConfigTripleClassification()
    else:
        raise ValueError("wrong class num")

    device = torch.device("cuda:%d" % args.cuda)
    Data = DataIteratorDistill(config=cfg, train_batchsize=args.batch_size)
    model = torch.load("checkpoints/CNN-29", map_location=device) 

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = FocalLoss(classes=args.class_num, device=device).to(device)
    criterion_kv = nn.KLDivLoss().to(device)

    alpha = 0.2
    T = 2
    for epoch in range(args.epoch_num):
        print(epoch)
        for sample in Data.train_iter:
            model.train()
            optimizer.zero_grad()
            output = model(sample.text.permute(1, 0).to(device))
            loss_f = criterion(output, sample.label.to(device))
            output = F.log_softmax(output/T, 1)
            score = torch.cat((sample.pred0.unsqueeze(1).to(device), 
                sample.pred1.unsqueeze(1).to(device)), dim=1)
            score = F.softmax(score/T,1)
            loss_kv = criterion_kv(output, score.to(device)) * T * T
            loss = alpha * loss_f + (1 - alpha) * loss_kv
            #print(loss_f.item(), loss_kv.item())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            preds = []
            labels = []
            for sample in Data.valid_iter:
                output = model(sample.text.permute(1, 0).to(device))
                p = output.argmax(1).cpu().tolist()
                l = sample.label.tolist()
                preds += p
                labels += l
            report = classification_report(preds, labels)
            print(report)
            torch.save(model, os.path.join(args.save_dir, args.save_config + str(epoch)))
