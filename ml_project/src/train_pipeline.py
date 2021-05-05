import sys
import json
import logging
import datetime

import data_load
import preprocessing
import train_model
import transformer

import configparser
import pandas as pd

logging.basicConfig(filename='../logs/logs_file.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

config = configparser.ConfigParser()
config.read('../configs/main_config.ini')

def train_pipeline():
    logging.debug("Train begining")
    
    data = data_load.read_data(config["train_config"]["data_path"])

    logging.debug("Data shape %s" % (data.shape, ))
    
    X = data.drop(config["train_config"]["target_name"], axis=1)
    y = data[config["train_config"]["target_name"]]

    tran = transformer.CustTransformer(X, int(config["preproc"]["n_stdev"]))
    X = tran.transform()

    X_train, X_test, y_train, y_test = data_load.train_test_split_conf(
        X, y,
        float(config["train_config"]["split"]), 
        int(config["train_config"]["random_seed"])
    )
    
    logging.debug("Train dataset: %s", (X_train.shape, ))
    logging.debug("Test dataset: %s", (X_test.shape, ))

    logit = train_model.train(X_train, y_train, "logit")
    y_pred_logit = train_model.predict(logit, X_test)
    roc_auc_logit = train_model.validate(y_pred_logit, y_test)

    rf = train_model.train(X_train, y_train, "randomforest")
    y_pred_rf = train_model.predict(rf, X_test)
    roc_auc_rf = train_model.validate(y_pred_rf, y_test)

    logging.debug("Train is success: logit roc-auc %s", (roc_auc_logit, ))

    train_model.save_model(logit, config["train_config"]["model_logit"])
    train_model.save_model(rf, config["train_config"]["model_rf"])

    logging.debug("Models saved")

    with open(config["train_config"]["metric_file"], "w") as metric_file:
        res = {
            "logit": {
                "time": str(datetime.datetime.now()),
                "roc_auc": roc_auc_logit[0]
            },
            "randomforest": {
                "time": str(datetime.datetime.now()),
                "roc_auc": roc_auc_rf[0]
            }

        }
        json.dump(res, metric_file)
        return res

if __name__ == "__main__":
    train_pipeline()