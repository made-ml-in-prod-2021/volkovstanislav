import os
import json
import datetime
import pandas as pd
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from src.transformer import CustTransformer

UPLOAD_FOLDER = 'data/upload_files'
ALLOWED_EXTENSIONS = set(['txt', 'csv'])
COLUMNS_LIST = ["age", "sex", "cp", "trestbps",
            "chol", "fbs", "restecg", "thalach", 
            "exang", "oldpeak", "slope", "ca", "thal"]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FILE'] = 'models/model_logit.pkl'
app.config['RESULT_FOLDER'] = 'data/predict_files'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def validate_data(data):
    data_problem = 0
    
    # Проверяем совпадение по столбцам
    data_columns = data.columns
    if len(set(data_columns) & set(COLUMNS_LIST)) != data.shape[1]:
        data_problem += 1

    # Проверяем что в данных нет отрицательных значений
    if data[data < 0].shape[0] != 0:
        data_problem += 1

    # Проверяем, что нет колонок с текстовыми значениями
    for col in data.columns:
        if(data[col].dtype == "object"):
            data_problem += 1

    return data_problem
    

@app.route('/')
def hello_world():
    return '''
    <h3> Hello, friend! If you want predict Potentional Heart Disease: <br />
    - from the file go to <a href="/predict">Predict from file</a> <br />
    File must be only CSV FORMAT 
    '''


@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File for Predict Heart Disease</h1>
    <form action="" method=post enctype=multipart/form-data>
    <p><input type=file name=file>
        <input type=submit value=Upload>
    </form>
    '''


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    X = pd.read_csv(app.config['UPLOAD_FOLDER'] + '/' + filename)
    data_check = validate_data(X)
    if data_check != 0:
        model = pd.read_pickle(app.config['MODEL_FILE'])

        tran = CustTransformer(X, 2)
        X = tran.transform()
        y_pred = model.predict_proba(X)

        return_y = []
        for i in range(len(y_pred)):
            return_y.append(y_pred[i][1])

        res = {
            'model_params': model.coef_.tolist()[0],
            'y_pred': return_y,
            'status': 'Success',
            "datetime": str(datetime.datetime.now(tz=None))
        }
        json.dump(res, open(app.config['RESULT_FOLDER'] + '/' + filename + ".json", 'w')) 

        return res
    else:
        return "Проблема с загруженными данными", 400


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')