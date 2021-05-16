import os

class ConfigBase(object):
    RAW_DATA_PATH = "./data/raw/"
    TRAIN_DATA_FILE = "train.tsv"
    VALID_DATA_FILE = "valid.tsv"
    UNSUP_DATA_FILE = "unsup.tsv"
    SPLIT_RANDOM_SEED = 0
    add_vocab_file = "data/add_vocab.txt"

class ConfigTripleClassification(ConfigBase):
    RAW_DATA_FILES = ["UC.xlsx", "CR.xlsx", "JH.xlsx"]
    RAW_DATA_FILES_UNSUP = ["train.xlsx"]
    DATA_PATH = "./data/UC-CR-JH/"
    label_map = {"UC": 0, "CR": 1, "JH": 2}


class ConfigBinaryClassificationDistill(ConfigBase):
    RAW_DATA_FILES = ["CR.xlsx", "JH.xlsx"]
    RAW_DATA_FILES_UNSUP = ["UC.xlsx", "train.xlsx"]
    DATA_PATH = "./data/CR-JH-distill/"
    label_map = {"CR": 0, "JH": 1}


class ConfigBinaryClassification(ConfigBase):
    RAW_DATA_FILES = ["CR.xlsx", "JH.xlsx"]
    RAW_DATA_FILES_UNSUP = ["UC.xlsx", "train.xlsx"]
    DATA_PATH = "./data/CR-JH/"
    label_map = {"CR": 0, "JH": 1}


def init(config):
    if not os.path.exists(config.DATA_PATH):
        os.makedirs(config.DATA_PATH)
