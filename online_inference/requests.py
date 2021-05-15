'''Модуль с запросами к сервису'''
import json
import random
import pandas as pd
from start import app
from io import BytesIO

MAX_CASE = 100
MIN_CASE = 5
REQUEST_DATA_PATH = "data/upload_files"

def generate_data_for_questions():
    num = random.randint(MIN_CASE, MAX_CASE)
    data = {
        'age': [random.randint(15, 100) for _ in range(num)],
        'sex': [random.randint(0, 1) for _ in range(num)],
        'cp': [random.randint(0, 3) for _ in range(num)],
        'trestbps': [random.randint(70, 120) for _ in range(num)],
        'chol': [random.randint(90, 250) for _ in range(num)],
        'fbs': [random.randint(0, 1) for _ in range(num)],
        'restecg': [random.randint(0, 1) for _ in range(num)],
        'thalach': [random.randint(140, 180) for _ in range(num)],
        'exang': [random.randint(0, 1) for _ in range(num)],
        'oldpeak': [random.uniform(0.4, 4.1) for _ in range(num)],
        'slope': [random.randint(0, 2) for _ in range(num)],
        'ca': [random.randint(0, 1) for _ in range(num)],
        'thal': [random.randint(0, 2) for _ in range(num)],
    }
    return pd.DataFrame(data)

def main():
    num = random.randint(MIN_CASE, MIN_CASE + 5)
    for i in range(num):
        data = generate_data_for_questions()
        url = REQUEST_DATA_PATH + "/test-" + str(i) + ".csv"
        data.to_csv(url, index=False)
        
        client = app.test_client()
        response = client.get('/uploads' + "/test-1.csv")
        print(response.data)
        res = json.loads(response.data)
        json.dump(res, open("data/request_data" + "/test-" + str(i) + ".json", 'w')) 
    
if __name__ == "__main__":
    main()
