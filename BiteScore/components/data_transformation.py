import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from BiteScore.Logging.logger import logger
from BiteScore.Exception.exception import BiteScoreException
from BiteScore.config.entity_config import DataTransformation
from BiteScore.pipeline.preprocessing import PreprocessingPipeline
from BiteScore.config.artifact_config import DataValidationArtifact, DataTransformationArtifact
from BiteScore.utils.functionalities import read_yaml, save_numpy_array_data, save_model

# Reading the config file
CONFIG = read_yaml(Path("config.yaml"))

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformation, data_validation_artifact: DataValidationArtifact):
        try:
            logger.info("Initializing the data transformation configuration")
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            logger.error("Error in initializing the data transformation configuration")
            raise BiteScoreException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            logger.info(f"Reading the data from {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error("Error in reading the data")
            raise BiteScoreException(e, sys)
    
    def build_transformer_object(self, train_data: pd.DataFrame):
        try:
            logger.info("Building transformer object")
            # Initialize the preprocessing pipeline object
            preprocessing_pipeline = PreprocessingPipeline(CONFIG, train_data)
            # Get the preprocessing pipeline
            pipeline = preprocessing_pipeline.get_pipeline()
            logger.info("Transformer object built successfully")
            return pipeline
        except Exception as e:
            logger.error("Error in building transformer object")
            raise BiteScoreException(e, sys)

    def transform_data(self):
        try:
            logger.info("Reading the data")
            # Read the data
            train_data = self.read_data(self.data_validation_artifact.train_data_path)
            test_data = self.read_data(self.data_validation_artifact.test_data_path)
            val_data = self.read_data(self.data_validation_artifact.val_data_path)

            # Training Dataframe
            train_data.drop(subset=['name','address'], axis=0, inplace=True)
            input_feature_train_df = train_data.drop(columns=[CONFIG.TARGET_COLUMN], axis=1)
            target_feature_train_df = train_data[CONFIG.TARGET_COLUMN]

            # Testing Dataframe
            test_data.dropna(subset=['name','address'], axis=0, inplace=True)
            input_feature_test_df = test_data.drop(columns=[CONFIG.TARGET_COLUMN], axis=1)
            target_feature_test_df = test_data[CONFIG.TARGET_COLUMN]

            # Validation Dataframe
            val_data.dropna(subset=['name','address'], axis=0, inplace=True)
            input_feature_val_df = val_data.drop(columns=[CONFIG.TARGET_COLUMN], axis=1)
            target_feature_val_df = val_data[CONFIG.TARGET_COLUMN]

            # Build the transformer object
            transformer = self.build_transformer_object(train_data)

            # Transform the data
            logger.info("Transforming the data")
            transformer.fit(input_feature_train_df)
            transformed_train_data = transformer.transform(input_feature_train_df)
            transformed_test_data = transformer.transform(input_feature_test_df)
            transformed_val_data = transformer.transform(input_feature_val_df)

            # Combine transformed features with the target feature
            train_arr = np.c_[transformed_train_data, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_test_data, np.array(target_feature_test_df)]
            val_arr = np.c_[transformed_val_data, np.array(target_feature_val_df)]

            # Save transformed data
            logger.info("Saving the transformed data")
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_val_file_path, array=val_arr)

            # Save the transformer model
            save_model(path=self.data_transformation_config.transformed_object_file_path, data=transformer)
            save_model(path="final_model/preprocessor.pkl", data=transformer)

            logger.info("Data transformation completed successfully")

            # Creating the DataTransformationArtifact object
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_val_file_path=self.data_transformation_config.transformed_val_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact
        except Exception as e:
            logger.error("Error during data transformation")
            raise BiteScoreException(e, sys)