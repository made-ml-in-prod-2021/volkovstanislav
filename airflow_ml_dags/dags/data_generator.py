import os
import random
import datetime
import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


def generate_data():
    #sample length
    n = random.randint(100, 1000)

    data = pd.DataFrame()
    #factors
    data['col1'] = abs(np.random.normal(1, 12, n))
    data['col2'] = abs(np.random.normal(2, 8, n))
    data['col3'] = abs(np.random.normal(3, 2, n))
    data['col4'] = abs(np.random.normal(10, 15, n))
    data['col5'] = abs(np.random.normal(10, 15, n))

    #target
    data['target'] = random.randint(0, 1)

    path = '/opt/airflow/data/raw/' + str(datetime.datetime.now().date())

    if os.path.isdir(path):
        os.chdir(path)
    else:
        os.mkdir(path)
        os.chdir(path)

    print('/opt/airflow/data/raw/{{ ds }}/data.csv')
    data.drop('target', axis=1).to_csv(path + '/data.csv', index=False)
    data['target'].to_csv(path + '/target.csv', index=False)


with DAG(
    dag_id="generate_data_and_save",
    start_date=datetime.datetime.now(),
    schedule_interval=None,
) as dag:

    create_directory = BashOperator(
        task_id="create_directory",
        bash_command='mkdir -p /opt/airflow/data/raw/{{ ds }}',
    )

    gen_data = PythonOperator(
        task_id="gen_data_and_save", python_callable=generate_data,
    )

    notify = BashOperator(
        task_id="notify",
        bash_command='echo "Generate data and save in ---/opt/airflow/data/raw/{{ ds }}/data.csv---"',
    )

    create_directory >> gen_data >> notify
