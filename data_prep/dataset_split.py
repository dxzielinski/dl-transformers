"""
Dataset preparation.
It assumes that whole dataset is not splitted into train and test sets.
It is assumed that the dataset is in the following structure:

/data_dir
        /class_1
        /class_2
        .
        .
        /class_n

As a result, the data-audio/ directory should appear in the project root directory
with a proper train, val and test split.
"""

import os
import shutil
import random

random.seed(111)
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
downloaded_data_dir = os.path.join(project_root_dir, "raw_audio")

labels = os.listdir(downloaded_data_dir)
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(project_root_dir, "data-audio")
TRAIN_PATH = os.path.join(PATH, "train")
VAL_PATH = os.path.join(PATH, "val")
TEST_PATH = os.path.join(PATH, "test")

if not os.path.exists(PATH):
    shutil.copytree(
        downloaded_data_dir,
        PATH,
    )

if (
    not os.path.exists(TRAIN_PATH)
    or not os.path.exists(VAL_PATH)
    or not os.path.exists(TEST_PATH)
):
    os.makedirs(TRAIN_PATH)
    os.makedirs(VAL_PATH)
    os.makedirs(TEST_PATH)
for label in labels:
    if not os.path.exists(os.path.join(TRAIN_PATH, label)):
        os.makedirs(os.path.join(TRAIN_PATH, label))
    if not os.path.exists(os.path.join(VAL_PATH, label)):
        os.makedirs(os.path.join(VAL_PATH, label))
    if not os.path.exists(os.path.join(TEST_PATH, label)):
        os.makedirs(os.path.join(TEST_PATH, label))

user_ids = set()
for label in labels:
    files = os.listdir(os.path.join(downloaded_data_dir, label))
    for file in files:
        user_id = file.split("_")[0]
        user_ids.add(user_id)
user_ids = list(user_ids)
random.shuffle(user_ids)
train_ids = user_ids[: int(len(user_ids) * 0.7)]
val_ids = user_ids[int(len(user_ids) * 0.7) : int(len(user_ids) * 0.85)]
test_ids = user_ids[int(len(user_ids) * 0.85) :]

for label in labels:
    files = os.listdir(os.path.join(downloaded_data_dir, label))
    train = [file for file in files if file.split("_")[0] in train_ids]
    val = [file for file in files if file.split("_")[0] in val_ids]
    test = [file for file in files if file.split("_")[0] in test_ids]
    for file in train:
        os.rename(
            os.path.join(PATH, label, file),
            os.path.join(TRAIN_PATH, label, file),
        )
    for file in val:
        os.rename(
            os.path.join(PATH, label, file),
            os.path.join(VAL_PATH, label, file),
        )
    for file in test:
        os.rename(
            os.path.join(PATH, label, file),
            os.path.join(TEST_PATH, label, file),
        )

for label in labels:
    os.rmdir(os.path.join(PATH, label))
