import pickle
import transformer
import configparser

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

config = configparser.ConfigParser()
config.read('../configs/main_config.ini')


def train(X_train, y_train, model_type):
    if model_type == "randomforest":
        model = RandomForestClassifier(
            n_estimators=int(config['randomforest']['n_estimators']),
            random_state=int(config['randomforest']['random_seed']),
        )
    elif model_type == "logit":
        model = LogisticRegression(
            C=int(config['logit']['const']),
            max_iter=int(config['logit']['max_iter'])
        )
    else:
        raise ValueError('Unknown type of model')
    
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred


def validate(y_pred, y_real):
    return roc_auc_score(y_real, y_pred),


def save_model(model, path):
    with open(path, "wb") as file:
        pickle.dump(model, file)
    return path


def load_model(path):
    with open(path, "rb") as file:
        model = pickle.load(file)
    return model


def predict_new(X):
    model = load_model(config["train_config"]["model_rf"])
    tran = transformer.CustTransformer(X, int(config["preproc"]["n_stdev"]))
    X = tran.transform()
    return predict(model, X)
     