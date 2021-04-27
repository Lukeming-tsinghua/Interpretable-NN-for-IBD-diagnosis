import os
from collections import namedtuple

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.optim import Adam
from tqdm import tqdm

from data import DataIterator
from loss import FocalLoss
from model import CNN
from torchtext import data, vocab

from args import get_args, print_args
from config import ConfigBinaryClassification
from config import ConfigTripleClassification

if __name__ == "__main__":
    args = get_args()
    print_args(args)

    if args.class_num == 2:
        cfg = ConfigBinaryClassification()
    elif args.class_num == 3:
        cfg = ConfigTripleClassification()
    else:
        raise ValueError("wrong class num")

    device = torch.device("cuda:%d" % args.cuda)
    Data = DataIterator(config=cfg, train_batchsize=args.batch_size)
    model = CNN(vocab_size=len(Data.vocab),
                embedding_dim=100,
                n_filters=100,
                filter_sizes=range(2, 5),
                output_dim=args.class_num,
                dropout=0.5,
                pad_idx=1).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = FocalLoss(classes=args.class_num, device=device).to(device)

    for epoch in range(args.epoch_num):
        print(epoch)
        for sample in Data.train_iter:
            model.train()
            optimizer.zero_grad()
            output = model(sample.text.permute(1, 0).to(device))
            loss = criterion(output, sample.label.to(device))
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
