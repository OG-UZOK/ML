from flask import Flask, jsonify, request
from catboost import CatBoostClassifier, Pool
import pandas as pd

app = Flask(__name__)

# �������� ������
model = CatBoostClassifier()  # ��������������� ������, ��� ��� � ��� �������� � ����� ����
model.load_model('model.bin')  # ���� � ����� ��������� ������

# ����� ��� ������������
@app.route('/predict', methods=['POST'])
def predict():
    # �������� ������ �� �������
    data = request.get_json()

    # ����������� ������ � DataFrame
    df = pd.DataFrame(data)

    # ������������� ������ (���� ����������)
    # ����� ������ ���� ����� �� ������������� ������, ��� � ����� ���� ��� �������� ������

    # ������������ � ������� ������
    predictions = model.predict(df)

    # ���������� ������������ � ������� JSON
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(port=5000)
