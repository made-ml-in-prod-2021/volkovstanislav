import logging
import data_load
import train_model
import pandas as pd
import configparser

logging.basicConfig(filename='../logs/logs_file.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

config = configparser.ConfigParser()
config.read('../configs/main_config.ini')

X = data_load.read_data(config["train_config"]["sample_path"])
logging.debug("Success load data for predict with shape: %s", (X.shape, ))

y_pred = train_model.predict_new(X)
logging.debug("Success load data for predict")

print("Predict of the new data:", y_pred)
