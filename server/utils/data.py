import torch
from LAC import LAC
from torchtext import data
from torchtext import vocab
from collections import namedtuple


class LacTokenizer(object):
    def __init__(self):
        self.lac = LAC(mode="seg")
        self.lac.load_customization('static/data/addwords.txt', sep="\t")

    def __call__(self, text):
        return self.lac.run(text)

    
def DataIterator(train="train.tsv", valid="valid.tsv", token="lac", padding_length=256,
        train_batchsize=128, valid_batchsize=128):
    if token == "lac":
        tokenizer = LacTokenizer()
    TEXT = data.Field(lower=True, tokenize=tokenizer, fix_length=padding_length)
    LABEL = data.Field(sequential=False, use_vocab=False)
    
    train_dataset, valid_dataset = data.TabularDataset.splits(
                path = "static/data/CR-JH/", format = 'tsv', skip_header = False,
                train=train, validation=valid,
                fields=[
                            ('text', TEXT),
                            ('label', LABEL)
                       ]
                )
    
    TEXT.build_vocab(train_dataset, valid_dataset)
    LABEL.build_vocab(train_dataset, valid_dataset)
    
    train_iter, valid_iter = data.Iterator.splits((train_dataset, valid_dataset), 
            batch_sizes = (train_batchsize, valid_batchsize),sort_key=lambda x: len(x.text))
    Data = namedtuple("data","train_iter valid_iter vocab tokenizer")
    return Data(train_iter=train_iter, valid_iter=valid_iter, vocab=TEXT.vocab, tokenizer=TEXT)


if __name__ == "__main__":
    Data = DataIterator()
    print(Data)
