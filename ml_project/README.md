# volkovstanislav
Machine Learning in Production Homework Repository

MADE link profile: https://data.mail.ru/profile/s.volkov/

### Create environment
```
git clone https://github.com/made-ml-in-prod-2021/volkovstanislav.git
cd ml_project/
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Train model
```
cd src 
python3 train_pipeline.py 
```

### Predict model
```
cd src
python3 predict_pipeline.py
```

### Test model pipeline
```
cd tests
pytest *
```

### Самоценка
- 2) сделано +1
- 1) сделано +0
0) сделано +2 
1) сделано +2
2) сделано +2
3) сделано +2
4) сделано частично +2 (можно было покрыть большим кол-вом тестов)
5) сделано +3 (не было идеи как применять faker или другие аналоги для генерации, поэтому тестовые примеры генерил своей функцией)
6) сделано частично +2 (использовалcя congfig, а не предложенный yaml)
7) не сделано (в конфигал были прописаны значения)
8) сделано частично +2 (простой трансоформер с простыми тестами)
9) сделано +3
10) сделано +2
11) не сделано +0
12) не сделано +0
13) сделано +1
====== Всего 24 балла


## Project Organization
```
├── README.md          <- The top-level README for developers using this project.
│
├── data               <- All data for train and validate models
│
├── models             <- Trained and serialized models.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│
├── logs               <- Logs file of Project interaction file
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment.
│
└── src                <- Source code for use in this project.
│
└── tests              <- unit tests
│
└── configs            <- Config file to interaction of this project
```
