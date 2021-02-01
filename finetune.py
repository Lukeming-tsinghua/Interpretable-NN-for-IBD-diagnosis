import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from transformers import BertTokenizer
from transformers import BertConfig
from transformers import BertPreTrainedModel
from transformers import BertForSequenceClassification
from transformers import BertModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from data import SentenceDataset
from loss import FocalLoss

from args import get_args, print_args
from config import ConfigBinaryClassification
from config import ConfigTripleClassification


def finetune(args, cfg):
    device = torch.device("cuda:%d" % args.cuda)
    model_config = args.model_config
    tokenizer = BertTokenizer.from_pretrained(model_config)
    train_dataset = SentenceDataset(tokenizer,cfg.DATA_PATH, dataset="train", cuda=False)
    valid_dataset = SentenceDataset(tokenizer,cfg.DATA_PATH, dataset="valid", cuda=False)

    train_loader = DataLoader(train_dataset, batch_size=16)
    valid_loader = DataLoader(valid_dataset, batch_size=16)

    model = BertForSequenceClassification.from_pretrained(model_config,num_labels=args.class_num)
    model.to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    #criterion = FocalLoss(classes=3, device=device).to(device)

    for epoch in range(args.epoch_num):
        for tokens, label in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()

            tokens = {key: item.to(device) for key, item in tokens.items()}
            label = label.to(device)

            pred = model(**tokens)[0]
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            tokens = {key: item.cpu() for key, item in tokens.items()}
            label = label.cpu()
            del tokens, label

        with torch.no_grad():
            model.eval()
            preds = []
            labels = []
            for tokens, label in tqdm(valid_loader):
                tokens = {key: item.to(device) for key, item in tokens.items()}
                pred = model(**tokens)[0]
                p = pred.argmax(1).cpu().tolist()
                l = label.tolist()
                preds += p
                labels += l
            report = classification_report(preds, labels)
            print(report)
            model.save_pretrained(os.path.join(args.save_dir, args.save_config+str(epoch)))


if __name__ == "__main__":
    args = get_args()
    print_args(args)

    if args.class_num == 2:
        cfg = ConfigBinaryClassification()
    elif args.class_num == 3:
        cfg = ConfigTripleClassification()
    else:
        raise ValueError("wrong class num")

    finetune(args, cfg)
