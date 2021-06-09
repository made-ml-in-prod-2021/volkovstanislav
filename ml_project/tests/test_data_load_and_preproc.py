import os
import sys
sys.path.insert(1, '../src')

import pytest
import data_load
import numpy as np
import pandas as pd

DATA_NAME = 'heart.csv'
DATA_SAMPLE_NAME = 'sample_heart.csv'
TEST_SAMPLE = "save_test_sample.csv"
DATA_FOLDER_PATH = '../data/'


def test_has_file_in_dir():
    path = DATA_FOLDER_PATH
    assert len(os.listdir(path)) > 0, (
        'Not file in directory'
    )
    assert os.path.isfile(os.path.join(path, DATA_NAME)), (
        'Not file in directory'
    )

    assert os.path.join(path, DATA_NAME) == DATA_FOLDER_PATH + DATA_NAME, (
        'Not file in directory'
    )


def test_read_csv_file():
    data = data_load.read_data(DATA_FOLDER_PATH + DATA_SAMPLE_NAME)
    age_vector = np.array([63, 37, 41, 56, 57])
    assert (data["age"].values == age_vector).all(), (
        "Problems with read file"
    )


def test_save_csv_file():
    data = {
        "id": [1, 3, 5],
        "name": ["Ivan", "Petr", "Yaroslav"]
    }
    data = pd.DataFrame(data)
    data_load.save_data(data, DATA_FOLDER_PATH + TEST_SAMPLE)
    load_data = data_load.read_data(DATA_FOLDER_PATH + TEST_SAMPLE)
    assert (data["id"].values == load_data["id"]).all(), (
        "Problems with save file"
    )

def test_sample_split():
    sample_size = [0.25, 0.3, 0.5]
    data = data_load.read_data(DATA_FOLDER_PATH + DATA_NAME)
    X = data.drop("target", axis=1)
    y = data["target"]

    for size in sample_size:
        X_train, X_test, y_train, y_test = data_load.train_test_split_conf(
            X, y,
            sample_size[0], 
            2021
        )
        assert round((X_test.shape[0]) / (X_train.shape[0] + X_test.shape[0]), 2) == sample_size[0], (
            "Problems with train test split function"
        )






