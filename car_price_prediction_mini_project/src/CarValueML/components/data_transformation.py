import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.CarValueML.exception import CustomException
from src.CarValueML.logger import logging
from src.CarValueML.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join("artifacts", "preprocessor_obj.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def create_preprocessor_obj(self):
        try:
            num_features = [
                "Year",
                "Distance",
                "Owner"
                ]

            cat_features = [
                "Fuel",
                "Drive",
                "Type",
                "State",
                "Brand",
                "Model Name"
                ]

            # Initializing the transformers
            num_transformer = StandardScaler()
            cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

            # creating the preprocessing object
            preprocessor = ColumnTransformer(
                [("num_transformer", num_transformer, num_features),
                 ("cat_transformer", cat_transformer, cat_features)]
                )

            logging.info("Preprocessor object created")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path: str, test_data_path: str):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Reading Training and Testing data")
            logging.info("Getting Preprocessor object")

            preprocessor_obj = self.create_preprocessor_obj()
            target_column = "Price(k)"

            logging.info("Splitting dataframes into dependent and independent variables")

            train_feature_df = train_df.drop(target_column, axis=1)
            train_target_df = train_df[target_column]

            test_feature_df = test_df.drop(target_column, axis=1)
            test_target_df = test_df[target_column]

            logging.info("Applying preprocessor object to training and testing data")

            input_features_train_arr = preprocessor_obj.fit_transform(train_feature_df)
            input_features_test_arr = preprocessor_obj.transform(test_feature_df)

            logging.info("Adding back the dependent feature")
            train_arr = np.c_[input_features_train_arr, np.array(train_target_df)]
            test_arr = np.c_[input_features_test_arr, np.array(test_target_df)]

            logging.info("Saving preprocessor object")
            save_object(
                self.data_transformation_config.preprocessor_obj_path,
                preprocessor_obj
                )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
                )

        except Exception as e:
            raise CustomException(e, sys)
