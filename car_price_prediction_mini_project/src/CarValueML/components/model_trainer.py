import os
import sys
import warnings
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.CarValueML.exception import CustomException
from src.CarValueML.utils import evaluate_models, save_object

warnings.filterwarnings("ignore")
from src.CarValueML.logger import logging


@dataclass
class ModelTrainerConfig:
    pretrained_model_path = os.path.join("artifacts", "pretrained_model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def inititate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Model Trainer initialized")
            logging.info("Reading training and testing data")

            x_train, y_train, x_test, y_test = train_arr[:, :-1], train_arr[:, -1], test_arr[:, :-1], test_arr[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Elastic Net": ElasticNet(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "Adaboost": AdaBoostRegressor(),
                "Decision tree": DecisionTreeRegressor(),
                "Catboost": CatBoostRegressor(verbose=0),
                "xgboost": XGBRegressor()
                }
            logging.info("Training and evaluating models")
            model_report = evaluate_models(x_train, y_train, x_test, y_test, models)

            # Getting the best model
            logging.info("Getting the best model")
            best_model_score = min(sorted(list(model_report.values())))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            logging.info(
                f"Best found model on both training and testing dataset: {best_model_name} with score: {best_model_score}")

            logging.info("Saving the best model")
            save_object(
                file_path=self.model_trainer_config.pretrained_model_path,
                obj=best_model
                )

        except Exception as e:
            raise CustomException(e, sys)
