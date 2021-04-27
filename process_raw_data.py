import argparse
import os

import sklearn.model_selection as model_selection
import xlrd

import config
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_num",
                        default=3,
                        type=int,
                        help="class number")
    args = parser.parse_args()

    if args.class_num == 2:
        cfg = config.ConfigBinaryClassification()
    elif args.class_num == 3:
        cfg = config.ConfigTripleClassification()
    else:
        raise ValueError("class number should be 2 or 3")
    config.init(cfg)

    # process train and test data
    x = []
    y = []
    for file_name in cfg.RAW_DATA_FILES:
        data_file_path = os.path.join(cfg.RAW_DATA_PATH, file_name)
        book = xlrd.open_workbook(data_file_path)
        sheet = book.sheet_by_index(0)
        rawdata = sheet.col_values(0)

        tmp_x = [utils.clean_text(line) for line in rawdata]
        tmp_y = [cfg.label_map[file_name.replace(".xlsx", "")]] * len(tmp_x)

        x += tmp_x
        y += tmp_y

    x_train, x_valid, y_train, y_valid = model_selection.train_test_split(
        x, y, test_size=0.2, random_state=cfg.SPLIT_RANDOM_SEED)

    data_format = "%s\t%d\n"

    with open(os.path.join(cfg.DATA_PATH, cfg.TRAIN_DATA_FILE), "w") as f:
        for data in zip(x_train, y_train):
            f.write(data_format % data)

    with open(os.path.join(cfg.DATA_PATH, cfg.VALID_DATA_FILE), "w") as f:
        for data in zip(x_valid, y_valid):
            f.write(data_format % data)

    # process unsup data
    x = []
    for file_name in cfg.RAW_DATA_FILES_UNSUP:
        data_file_path = os.path.join(cfg.RAW_DATA_PATH, file_name)
        book = xlrd.open_workbook(data_file_path)
        sheet = book.sheet_by_index(0)
        rawdata = sheet.col_values(0)

        x += [
            utils.clean_text(line) for idx, line in enumerate(rawdata)
            if idx != 0
        ]

    data_format = "%s\n"
    with open(os.path.join(cfg.DATA_PATH, cfg.UNSUP_DATA_FILE), "w") as f:
        for data in x:
            if "胃" not in data and "十二指肠" not in data:
                f.write(data_format % data)
