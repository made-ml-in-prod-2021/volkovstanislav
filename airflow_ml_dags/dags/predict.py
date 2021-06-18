import os
import json
import pickle
import datetime
import numpy as np
import pandas as pd

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


def predict_data():
    path_raw = '/opt/airflow/data/raw/' + str(datetime.datetime.now().date()) + '/data.csv'
    data = pd.read_csv(path_raw)

    path_predict = '/opt/airflow/data/predictions/' + str(datetime.datetime.now().date())
    if os.path.isdir(path_predict):
        os.chdir(path_predict)
    else:
        os.mkdir(path_predict)
        os.chdir(path_predict)

    model_path = '/opt/airflow/data/models/' + str(datetime.datetime.now().date())
    model_name = '/rf_' + str(datetime.datetime.now().date()) + '.pkl'  

    with open(model_path + model_name, 'rb') as file:  
        clf = pickle.load(file)

    y_pred = clf.predict(data)

    save_path = '/opt/airflow/data/predictions/' + str(datetime.datetime.now().date())
    save_name = '/predictions.csv'
    pd.DataFrame({'Predictions': y_pred}).to_csv(save_path + save_name, index=False)


with DAG(
    dag_id="predict",
    start_date=datetime.datetime.now(),
    schedule_interval="@daily",
) as dag:

    pred_data = PythonOperator(
        task_id="predict", python_callable=predict_data,
    )

    notify = BashOperator(
        task_id="notify",
        bash_command='echo "data predict and result save"',
    )

    pred_data >> notify
