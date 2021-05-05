import logging
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler

corr_threshold = 0.7

def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

    return dataset

def preprocessing(data):
    '''data preprocessing'''
    
    print("---------------------------------")
    print("Конвертация категориальных факторов - начало")
    logging.debug("Конвертация категориальных факторов - начало")
    data.sex = data.sex.astype('category')
    data.cp = data.cp.astype('category')
    data.fbs = data.fbs.astype('category')
    data.restecg = data.restecg.astype('category')
    data.exang = data.exang.astype('category')
    data.ca = data.ca.astype('category')
    data.slope = data.slope.astype('category')
    data.thal = data.thal.astype('category')
    print("Конвертация категориальных факторов - конец")
    logging.debug("Конвертация категориальных факторов - начало")
    print("---------------------------------")
    
    print("Создание dummy факторов на основе категориальных фич - начало")
    logging.debug("Создание dummy факторов на оснвое категориальных фич - начало")
    data = pd.get_dummies(data, drop_first=True)
    print("Создание dummy факторов на основе категориальных фич - конец")
    logging.debug("Создание dummy факторов на оснвое категориальных фич - конец")
    
    print("До корреляционного анализа:", data.shape)
    data = correlation(data, corr_threshold)
    print("После корреляционного анализа:", data.shape)
    print("---------------------------------")
    
    print("Нормализация данных - начало")
    logging.debug("Нормализация данных - начало")
    data_scaled = MinMaxScaler().fit_transform(data)
    data_scaled = pd.DataFrame(data=data_scaled, columns=data.columns)
    print("Нормализация данных - конец")
    logging.debug("Нормализация данных - конец")
    print("---------------------------------")
    return data_scaled
    
    
