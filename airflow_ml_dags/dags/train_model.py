import os
import json
import pickle
import datetime
import numpy as np
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def prepare_data():
    path_raw_data = '/opt/airflow/data/raw/' + str(datetime.datetime.now().date()) + '/data.csv'
    data = pd.read_csv(path_raw_data)

    path_raw_target = '/opt/airflow/data/raw/' + str(datetime.datetime.now().date()) + '/target.csv'
    target = pd.read_csv(path_raw_target)

    path_proc = '/opt/airflow/data/processed/' + str(datetime.datetime.now().date())
    if os.path.isdir(path_proc):
        os.chdir(path_proc)
    else:
        os.mkdir(path_proc)
        os.chdir(path_proc)

    train_data = pd.concat([data, target], axis=1)
    train_data.to_csv(path_proc + '/prep_data.csv', index=False)


def train_val_split():
    path_proc = '/opt/airflow/data/processed/' + str(datetime.datetime.now().date())

    data = pd.read_csv(path_proc + '/prep_data.csv')
    X = data.drop('target', axis=1)
    y = data['target']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
    X_train.to_csv(path_proc + '/X_train.csv', index=False)
    X_val.to_csv(path_proc + '/X_val.csv', index=False)
    y_train.to_csv(path_proc + '/y_train.csv', index=False)
    y_val.to_csv(path_proc + '/y_val.csv', index=False)


def train_model():
    path_proc = '/opt/airflow/data/processed/' + str(datetime.datetime.now().date())
    X_train = pd.read_csv(path_proc + '/X_train.csv')
    y_train = pd.read_csv(path_proc + '/y_train.csv')

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    model_name = '/rf_' + str(datetime.datetime.now().date()) + '.pkl'  

    model_path = '/opt/airflow/data/models/' + str(datetime.datetime.now().date())
    if os.path.isdir(model_path):
        os.chdir(model_path)
    else:
        os.mkdir(model_path)
        os.chdir(model_path)

    with open(model_path + model_name, 'wb') as file:  
        pickle.dump(clf, file)


def valid_model():
    path_proc = '/opt/airflow/data/processed/' + str(datetime.datetime.now().date())
    X_val = pd.read_csv(path_proc + '/X_val.csv')
    y_val = pd.read_csv(path_proc + '/y_val.csv')

    model_path = '/opt/airflow/data/models/' + str(datetime.datetime.now().date())
    model_name = '/rf_' + str(datetime.datetime.now().date()) + '.pkl'  

    with open(model_path + model_name, 'rb') as file:  
        clf = pickle.load(file)

    y_pred = clf.predict(X_val)
    res = {
        'model': 'random_forest',
        'score': accuracy_score(y_val, y_pred),
    }

    metric_path = '/opt/airflow/data/models/'
    metric_name = '/metric' + str(datetime.datetime.now().date()) + '.json'
    with open(metric_path + metric_name, "w") as outfile: 
        json.dump(res, outfile)


with DAG(
    dag_id="train_model",
    start_date=datetime.datetime.now(),
    schedule_interval="@weekly",
) as dag:

    prep_data = PythonOperator(
        task_id="prepare_data", python_callable=prepare_data,
    )

    train_val_splitting = PythonOperator(
        task_id="train_test_split", python_callable=train_val_split,
    )

    train_model = PythonOperator(
        task_id="train_model", python_callable=train_model,
    )

    valid_model = PythonOperator(
        task_id="valid_model", python_callable=valid_model,
    )

    prep_data >> train_val_splitting >> train_model >> valid_model
