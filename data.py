from collections import namedtuple

import torch

from config import ConfigBinaryClassification
from config import ConfigTripleClassification
from LAC import LAC
from torchtext import data, vocab


class LacTokenizer(object):
    def __init__(self, config):
        self.lac = LAC(mode="seg")
        self.lac.load_customization(config.add_vocab_file, sep="\t")

    def __call__(self, text):
        return self.lac.run(text)


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


if __name__ == "__main__":
    Data = DataIterator(ConfigBinaryClassification)
    print(Data)
