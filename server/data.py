from collections import namedtuple
import os
import warnings

import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

from config import ConfigBinaryClassification
from config import ConfigBinaryClassificationDistill
from config import ConfigTripleClassification
from LAC import LAC
from torchtext import data, vocab


warnings.filterwarnings("ignore")


class LacTokenizer(object):
    def __init__(self, config):
        self.lac = LAC(mode="seg")
        self.lac.load_customization(config.add_vocab_file, sep="\t")

    def __call__(self, text):
        return self.lac.run(text)


def DataIteratorDistill(config,
                 token="lac",
                 padding_length=256,
                 train_batchsize=128,
                 valid_batchsize=128):

    if token == "lac":
        tokenizer = LacTokenizer(config)

    TEXT = data.Field(lower=True,
                      tokenize=tokenizer,
                      fix_length=padding_length)
    LABEL = data.Field(sequential=False, use_vocab=False)
    PRED0 = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
    PRED1 = data.Field(sequential=False, use_vocab=False, dtype=torch.float)

    train_dataset, valid_dataset = data.TabularDataset.splits(
        path=config.DATA_PATH,
        format='tsv',
        skip_header=False,
        train=config.TRAIN_DATA_FILE,
        validation=config.VALID_DATA_FILE,
        fields=[('text', TEXT), ('label', LABEL), ('pred0', PRED0), ('pred1', PRED1)])

    TEXT.build_vocab(train_dataset, valid_dataset)
    LABEL.build_vocab(train_dataset, valid_dataset)
    PRED0.build_vocab(train_dataset, valid_dataset)
    PRED1.build_vocab(train_dataset, valid_dataset)

    train_iter, valid_iter = data.Iterator.splits(
        (train_dataset, valid_dataset),
        batch_sizes=(train_batchsize, valid_batchsize),
        sort_key=lambda x: len(x.text))
    Data = namedtuple("data", "train_iter valid_iter vocab tokenizer")
    return Data(train_iter=train_iter,
                valid_iter=valid_iter,
                vocab=TEXT.vocab,
                tokenizer=TEXT)


def DataIterator(config,
                 token="lac",
                 padding_length=256,
                 train_batchsize=128,
                 valid_batchsize=128):

    if token == "lac":
        tokenizer = LacTokenizer(config)

    TEXT = data.Field(lower=True,
                      tokenize=tokenizer,
                      fix_length=padding_length)
    LABEL = data.Field(sequential=False, use_vocab=False)

    train_dataset, valid_dataset = data.TabularDataset.splits(
        path=config.DATA_PATH,
        format='tsv',
        skip_header=False,
        train=config.TRAIN_DATA_FILE,
        validation=config.VALID_DATA_FILE,
        fields=[('text', TEXT), ('label', LABEL)])

    TEXT.build_vocab(train_dataset, valid_dataset)
    LABEL.build_vocab(train_dataset, valid_dataset)

    train_iter, valid_iter = data.Iterator.splits(
        (train_dataset, valid_dataset),
        batch_sizes=(train_batchsize, valid_batchsize),
        sort_key=lambda x: len(x.text))
    Data = namedtuple("data", "train_iter valid_iter vocab tokenizer")
    return Data(train_iter=train_iter,
                valid_iter=valid_iter,
                vocab=TEXT.vocab,
                tokenizer=TEXT)


class SentenceDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, path, max_length = 256, dataset="train", filetype=".tsv", cuda=False):
        self.tokenizer = tokenizer
        self.path = path
        self.max_length = max_length
        self.dataset = dataset
        self.filetype = filetype
        self.cuda = cuda
        self.data = None
        self.label = None
        self._load_data()

    def _load_data(self):
        assert self.dataset in ("train", "valid")
        filepath = os.path.join(self.path, self.dataset + self.filetype)
        texts = []
        labels = []
        with open(filepath) as f:
            line = f.readline()
            while line:
                text, label = line.replace("\n", "").split("\t")
                texts.append(text)
                labels.append(int(label))
                line = f.readline()
        self.data = self.tokenizer(texts, truncation=True, padding=True, max_length=256)
        self.labels = torch.tensor(labels)


    def __getitem__(self, idx):
        encodings = {key: torch.tensor(value[idx]).cuda() if self.cuda else torch.tensor(value[idx]) for key, value in self.data.items()}
        label = self.labels[idx].cuda() if self.cuda else self.labels[idx]
        return encodings, label

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    Data = DataIterator(ConfigBinaryClassificationDistill)
    print(Data)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModelWithLMHead.from_pretrained("bert-base-chinese")
    dataset = SentenceDataset(tokenizer,"./data/CR-JH/", cuda=True)
    print(dataset[0])


