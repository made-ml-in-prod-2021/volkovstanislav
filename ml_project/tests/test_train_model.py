import os
import sys
sys.path.insert(1, '../src')

import pytest
import numpy as np
import pandas as pd
import train_model

MODEL_PATH = "../models/"

def test_model_pickle_in_folder():
    assert len(os.listdir(MODEL_PATH)) > 0, (
        "There are no pickle of models"
    )



