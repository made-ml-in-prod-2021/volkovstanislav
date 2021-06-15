## Основные команды образа
```
FROM python:3.8-slim-buster
COPY . /app
WORKDIR /app 
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python", "start.py"]
```

## Собираем Docker образ 
```
docker image build -t ctacukoc/mlprod:1.0 .
```

## Запускаем Docker образ
```
docker run -it ctacukoc/mlprod:1.0 
```

## Оптимизация
Функциональность нашего приложения в полной мере не позволяет провести полноценный процесс оптимизации docker контейнера, но благодаря файлу .dockerignore удалось не загружать ненужную информацию в контейнер. Из-за этого удалось сократить место образа с 865.47 мб до 525.6 мб

## Публикуем образ в dockerhub
```
docker push ctacukoc/mlprod:1.0 
```

## Корректные команды docker pull/run
```
docker pull ctacukoc/mlprod:1.0
docker run -it ctacukoc/mlprod:1.0 
```

## Автоматическая генерация тестов для запросов к серверу
```
python requests.py
```

## Тестирование сервиса
```
pytest -v test_flask.py
```

## Самоценка
1 - сделано +3
2 - сделано +3
3 - сделано +2
4 - сдеално +3
5 - сделано +4
6 - сделано +2 (минус балл за примитивность оптимизации, но с другой стороны и приложение у нас простое)
7 - сделано +2 (https://hub.docker.com/r/ctacukoc/mlprod/)
8 - сделано +1
9 - самооценка +1
Итого: 21