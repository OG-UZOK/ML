!pip install catboost
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import KFold

# Функция для кросс-валидации
def k_fold(X, y, model, k=10, random_state=6):
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = []
    predictions = pd.DataFrame(columns=['PassengerId', 'Transported'])  # DataFrame для хранения прогнозов модели
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        try:
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            score = model.score(X_val, y_val)
            scores.append(score)

            predictions_subset = pd.DataFrame(columns=["PassengerId", 'Transported'])  # Создаем пустой DataFrame
        except Exception as e:
            print(f"Failed to fit the model: {e}")
    return np.mean(scores), predictions

# Чтение данных
train_df = pd.read_csv('train.csv')
features_df = pd.read_csv('test.csv')
targets_df = pd.read_csv('sample_submission.csv')
test_df = pd.merge(features_df, targets_df, how='inner', on='PassengerId')
merged_df = pd.concat([train_df, test_df], ignore_index=True)
dfs = [train_df, test_df, merged_df]

# Заполнение пропусков и преобразование категориальных признаков
mode_features = ['HomePlanet', 'Destination', 'VIP']
median_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cryo_sleep_depending_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

for df in dfs:
    for i in range(len(df)):
        if df.at[i, 'CryoSleep'] == True:
            for feature in cryo_sleep_depending_features:
                df.at[i, feature] = 0.0

    for f in mode_features:
        df[f] = df[f].fillna(df[f].mode().iloc[0])

    for i in range(len(df)):
        if pd.isnull(df.at[i, 'CryoSleep']):
            if any(df.at[i, feature] > 0 for feature in cryo_sleep_depending_features):
                df.at[i, 'CryoSleep'] = False
            else:
                df.at[i, 'CryoSleep'] = True

    for f in median_features:
        df[f] = df[f].fillna(df[f].median())

# Кодирование категориальных признаков
bin_feats = ['CryoSleep', 'VIP', 'Transported']
cat_feats = ['HomePlanet', 'Destination']

for f in bin_feats:
    map_dict = {value: i for i, value in enumerate(set(merged_df[f]))}
    for df in dfs:
        df[f] = df[f].map(map_dict)

for f in cat_feats:
    values = set(merged_df[f])
    for v in values:
        for df in dfs:
            df[f + '_' + v] = df[f] == v
    train_df = train_df.drop(columns=f)
    test_df = test_df.drop(columns=f)
    merged_df = merged_df.drop(columns=f)

uselessFeatures = ['PassengerId', 'Cabin', 'Name']
target = 'Transported'
X_train = train_df.drop(columns=target).drop(columns=uselessFeatures).values
y_train = train_df[target].values
X_test = test_df.drop(columns=target).drop(columns=uselessFeatures).values
y_test = test_df[target].values
X_merged = merged_df.drop(columns=target).drop(columns=uselessFeatures).values
y_merged = merged_df[target].values

# Обучение и оценка модели CatBoostClassifier
rf_model = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.1, loss_function='Logloss', logging_level='Silent')
score, predictions = k_fold(X_train, y_train, rf_model)
print(f'CatBoost accuracy: {score}')

pred = rf_model.predict(X_test)
pred = pred.astype(bool)

output = pd.DataFrame({'PassengerId': targets_df['PassengerId'],'Transported': pred})
output.to_csv('predictions.csv', index=False)

# Сохраняем прогнозы модели в CSV файл
# predictions.to_csv('predictions_catboost.csv', index=False)
