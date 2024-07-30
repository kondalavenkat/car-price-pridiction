import os
import pickle
import sys

import numpy as np
from sklearn.metrics import mean_squared_error

from src.CarValueML.exception import CustomException
from src.CarValueML.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)

    except Exception as e:
        raise CustomException(e, sys)


def rmse(true, predicted):
    return np.sqrt(mean_squared_error(true, predicted))


def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        report = {}

        for i in range(len(models)):
            logging.info(f"Training model: {list(models.keys())[i]}")
            # Getting the model
            model = list(models.values())[i]

            # Fitting the model
            model.fit(x_train, y_train)

            # making predictions
            train_preds = model.predict(x_train)
            test_preds = model.predict(x_test)

            # Creating model report
            train_rmse = rmse(y_train, train_preds)
            test_rmse = rmse(y_test, test_preds)

            # Adding the details to the model report
            report[list(models.keys())[i]] = test_rmse

        return report

    except Exception as e:
        raise CustomException(e, sys)
