import os
import json
from start import app
from flask import Flask, url_for
from io import BytesIO

FILE_FOR_PREDICT_PATH = "sample_heart.csv"
SAMPLE_FOR_TEST_PATH = "data_sample_to_save.csv"


FILE_CONTENT = b'''age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
63,1,3,145,233,1,0,150,0,2.3,0,0,1
37,1,2,130,250,0,1,187,0,3.5,0,0,2
41,0,1,130,204,0,0,172,0,1.4,2,0,2
56,1,1,120,236,0,1,178,0,0.8,2,0,2
57,0,0,120,354,0,1,163,1,0.6,2,0,2'''


def test_base_route():
    client = app.test_client()
    url = '/'
    response = client.get(url)
    assert response.status_code == 200


def test_predict_route():
    client = app.test_client()
    url = '/predict'
    response = client.get(url)
    assert response.status_code == 200


def test_file_upload():
    client = app.test_client()

    data = {
        'field': 'value',
        'file': (BytesIO(FILE_CONTENT), SAMPLE_FOR_TEST_PATH)
    }

    response = client.post('/predict', buffered=True,
                     content_type='multipart/form-data',
                     data=data)
    
    assert response.status_code == 302


def test_file_upload_path():
    client = app.test_client()

    data = {
        'field': 'value',
        'file': (BytesIO(FILE_CONTENT), SAMPLE_FOR_TEST_PATH)
    }

    response = client.post('/predict', buffered=True,
                     content_type='multipart/form-data',
                     data=data)
    
    assert SAMPLE_FOR_TEST_PATH in os.listdir('data/upload_files')


def test_predict_after_upload():
    client = app.test_client()
    url = '/uploads/' + SAMPLE_FOR_TEST_PATH
    response = client.get(url)
    res = json.loads(response.data)

    assert res['y_pred'] == [
        0.9304217057857966,
        0.8427028156206432,
        0.9604197588832033,
        0.9546241075036923,
        0.7281576008831059
    ]


def test_predict_params():
    client = app.test_client()
    url = '/uploads/' + FILE_FOR_PREDICT_PATH
    response = client.get(url)
    res = json.loads(response.data)

    assert res['model_params'] == [
        0.002997110664294159,
        -1.1667226955598753,
        0.7545883365054613,
        -0.01612684790216278,
        -0.0075565673906548576,
        0.03335390290840844,
        0.7477651483840452,
        0.024234037204835898,
        -1.2764658407449287,
        -0.2773910538789466,
        0.6337962717884627,
        -1.252560008114491,
        -1.4738370392600575
    ]


def test_predict_status():
    client = app.test_client()
    url = '/uploads/' + FILE_FOR_PREDICT_PATH
    response = client.get(url)
    res = json.loads(response.data)

    assert res['status'] == "Success"


def test_predict_result():
    client = app.test_client()
    url = '/uploads/' + FILE_FOR_PREDICT_PATH
    response = client.get(url)
    res = json.loads(response.data)

    assert res['y_pred'] == [
        0.9304217057857966,
        0.8427028156206432,
        0.9604197588832033,
        0.9561658968971899,
        0.7281576008831059
    ]