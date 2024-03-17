from flask import Flask, jsonify, request
from catboost import CatBoostClassifier, Pool
import pandas as pd

app = Flask(__name__)

# Загрузка модели
model = CatBoostClassifier()  # Инициализируйте модель, как это у вас делается в вашем коде
model.load_model('model.bin')  # Путь к вашей обученной модели

# Метод для предсказания
@app.route('/predict', methods=['POST'])
def predict():
    # Получаем данные из запроса
    data = request.get_json()

    # Преобразуем данные в DataFrame
    df = pd.DataFrame(data)

    # Предобработка данных (если необходимо)
    # Здесь должна быть такая же предобработка данных, как в вашем коде для обучения модели

    # Предсказание с помощью модели
    predictions = model.predict(df)

    # Возвращаем предсказания в формате JSON
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(port=5000)
