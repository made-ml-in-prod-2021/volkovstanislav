import os
import sys
sys.path.insert(1, '../src')

import pytest
import transformer
import configparser
import numpy as np
import pandas as pd

config = configparser.ConfigParser()
config.read('../configs/main_config.ini')

def test_custom_transform():
    original = {
        "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0],
        "col2": [0.1, 0.5, 50, 0.5, 0.4, 0.5, 0.9],
        "col3": [5.0, 3.0, 7.0, 9.0, 4.0, 9.0, -5.0]
    }
    original = pd.DataFrame(original)
    test = original.copy()

    test.at[6, 'col1'] = test["col1"].mean()
    test.at[2, 'col2'] = test["col2"].mean()
    test.at[6, 'col3'] = test["col3"].mean()

    tran = transformer.CustTransformer(original, int(config["preproc"]["n_stdev"]))
    custom = tran.transform()
    
    assert test.equals(custom), (
        "Problems with custom transformer"
    )

def generate(median=140, err=12, outlier_err=20000, size=9, outlier_size=1):
    '''
        generate data with outliers:
        just generate three parts of the data independently: first non-outliers, 
        then lower and upper outliers, merge them together
    '''
    errs = err * np.random.rand(size) * np.random.choice((-1, 1), size)
    data = median + errs

    upper_errs = outlier_err * np.random.rand(outlier_size)
    upper_outliers = median + err + upper_errs

    data = np.concatenate((data, upper_outliers))

    return data.tolist()

def test_custom_tranform_with_generate_data():
    n_iteration = 100
    for i in range(n_iteration):
        original = {
            "col1": generate(),
            "col2": generate(),
            "col3": generate()
        }
        original = pd.DataFrame(original)
        test = original.copy()

        mean_1 = test["col1"].mean()
        mean_2 = test["col2"].mean()
        mean_3 = test["col3"].mean()

        test.at[original.shape[0] - 1, 'col1'] = mean_1
        test.at[original.shape[0] - 1, 'col2'] = mean_2 
        test.at[original.shape[0] - 1, 'col3'] = mean_3

        tran = transformer.CustTransformer(original, int(config["preproc"]["n_stdev"]))
        custom = tran.transform()

        print(original) 
        print(test)
        print(custom) 

        assert test.equals(custom), (
            "Problems with custom transformer on iteration " + str(i)
        )