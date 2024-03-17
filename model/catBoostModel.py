from catboost import CatBoostClassifier

class CatBoostModel:
    def __init__(self, iterations=1000, depth=6, learning_rate=0.1, loss_function='Logloss', logging_level='Silent'):
        self.model = CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=learning_rate, loss_function=loss_function, logging_level=logging_level)
