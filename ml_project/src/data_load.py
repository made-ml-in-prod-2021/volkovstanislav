import os
import pandas as pd
from sklearn.model_selection import train_test_split

def read_data(path):
    data = pd.read_csv(path)
    return data

def save_data(data, path):
    data.to_csv(path)
    return data

def train_test_split_conf(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test