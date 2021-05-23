import os
import sys
sys.path.insert(1, '../src')

import pytest
import numpy as np
import pandas as pd
import train_pipeline

def test_train_pipeline_check():
    res = train_pipeline.train_pipeline()
    assert ((res["logit"]["roc_auc"] > 0.5) & (res["randomforest"]["roc_auc"] > 0.5)), (
        "Problem of training model, roc-auc equal 0.5"
    )
