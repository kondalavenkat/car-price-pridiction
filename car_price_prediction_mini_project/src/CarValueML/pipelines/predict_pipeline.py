import sys
import pandas as pd
from src.CarValueML.components.data_transformation import DataTransformationConfig
from src.CarValueML.components.model_trainer import ModelTrainerConfig
from src.CarValueML.exception import CustomException
from src.CarValueML.logger import logging
from src.CarValueML.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            preprocessor_obj_path = DataTransformationConfig().preprocessor_obj_path
            pretrained_model_path = ModelTrainerConfig().pretrained_model_path
            logging.info("Loading preprocessor object and pretrained model")
            preprocessor_obj = load_object(preprocessor_obj_path)
            pretrained_model = load_object(pretrained_model_path)
            logging.info("Loading completed")
            logging.info("Generating predictions...")
            data_transformed = preprocessor_obj.transform(features)
            prediction = pretrained_model.predict(data_transformed)
            return prediction[0]
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
            self,
            year: int,
            distance: int,
            state: str,
            brand: str,
            model: str,
            type: str,
            owner: int,
            fuel: str,
            drive: str
            ):
        self.year = year
        self.distance = distance
        self.state = state
        self.brand = brand
        self.model = model
        self.type = type
        self.owner = owner
        self.fuel = fuel
        self.drive = drive
        
    def get_data_as_dataframe(self):
        try:
            custom_data_dict = {
                "Year": [self.year],
                "Distance": [self.distance],
                "State": [self.state],
                "Brand": [self.brand],
                "Type": [self.type],
                "Owner": [self.owner],
                "Fuel": [self.fuel],
                "Drive": [self.drive],
                "Model Name": [self.model]
                }
            return pd.DataFrame(custom_data_dict)
        except Exception as e:
            raise CustomException(e, sys)
