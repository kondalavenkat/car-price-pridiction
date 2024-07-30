import sys

from src.CarValueML.components.data_ingestion import DataIngestion
from src.CarValueML.components.data_transformation import DataTransformation
from src.CarValueML.exception import CustomException
from src.CarValueML.components.model_trainer import ModelTrainer


def main():
    data_ingestion = DataIngestion()
    train_set, test_set = data_ingestion.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_set, test_set)
    model_trainer = ModelTrainer()
    model_trainer.inititate_model_trainer(train_arr, test_arr)


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        raise CustomException(e, sys)
