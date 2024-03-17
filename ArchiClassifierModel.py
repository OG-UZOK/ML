import os
import pandas as pd
import logging
import numpy as np
from catboost import CatBoostClassifier
import joblib

class ArchiClassifierModel:

    def train(self, dataset_filename):
        logging.info("Training the model...")
        # Load the dataset
        data = pd.read_csv(dataset_filename)
        X = data.drop(columns=['Transported'])
        y = data['Transported']
        # Train the model
        model = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.1, loss_function='Logloss', logging_level='Silent')
        model.fit(X, y)
        # Save the trained model
        model_dir = './model'
        os.makedirs(model_dir, exist_ok=True)
        model.save_model(os.path.join(model_dir, 'trained_model.bin'))
        logging.info("Model training completed.")

    def predict(self, dataset_filename):
        logging.info("Loading the trained model...")
        # Load the trained model
        model_path = './model/trained_model.bin'  # corrected path
        if not os.path.exists(model_path):
            logging.error("Trained model not found.")
            return
        model = CatBoostClassifier()
        model.load_model(model_path)
        # Load the dataset for prediction
        data = pd.read_csv(dataset_filename)
        # Make predictions
        predictions = model.predict(data)
        # Save predictions to CSV
        results_dir = './data'
        os.makedirs(results_dir, exist_ok=True)
        results_filename = os.path.join(results_dir, 'results.csv')
        pd.DataFrame({'Predictions': predictions}).to_csv(results_filename, index=False)
        logging.info(f"Predictions saved to {results_filename}.")
