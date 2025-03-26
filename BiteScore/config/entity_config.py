import os
from datetime import datetime
from pathlib import Path
from BiteScore.utils.functionalities import read_yaml

# Loading the configuration file
CONFIG = read_yaml(Path("../config.yaml"))


class TrainingPipeline:
    def __init__ (self, timestamp=datetime.now()):
        timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name=CONFIG.PIPELINE_NAME
        self.artifact_name=CONFIG.ARTIFACTS_DIR
        self.artifact_dir=os.path.join(self.artifact_name,timestamp)
        self.model_dir=os.path.join("final_model")
        self.timestamp: str=timestamp

class DataIngestion:
    # All the paths that are required for data ingestion
    def __init__ (self, training_pipeline_config:TrainingPipeline):
        self.data_ingestion_dir:str=os.path.join(
            training_pipeline_config.artifact_dir,"DataIngestion"
        )
        self.training_file_path: str = os.path.join(
                self.data_ingestion_dir, "train.csv"
            )
        self.testing_file_path: str = os.path.join(
                self.data_ingestion_dir, "test.csv"
            )
        self.validation_file_path : str = os.path.join(
                self.data_ingestion_dir, "val.csv"
        )
        self.collection_name_train: str = CONFIG.Database_Details.Train_Collection
        self.collection_name_test: str = CONFIG.Database_Details.Test_Collection
        self.collection_name_val: str = CONFIG.Database_Details.Validation_Collection
        self.database_name: str = CONFIG.Database_Details.Database_Name

class DataValidation:
    def __init__ (self, training_pipeline_config:TrainingPipeline):
        self.data_ingestion_dir:str=os.path.join(
            training_pipeline_config.artifact_dir,"DataValidation"
        )
        self.valid_data_dir: str = os.path.join(self.data_validation_dir, "Valid")
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir, "Invalid")
        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, "train.csv")
        self.valid_test_file_path: str = os.path.join(self.valid_data_dir, "test.csv")
        self.valid_val_file_path: str = os.path.join(self.valid_data_dir, "val.csv")
        self.invalid_train_file_path: str = os.path.join(self.invalid_data_dir, "train.csv")
        self.invalid_test_file_path: str = os.path.join(self.invalid_data_dir, "test.csv")
        self.invalid_val_file_path: str = os.path.join(self.invalid_data_dir, "val.csv")
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            "Report",
            "report.yaml",
        )

class DataTransformation:
    def __init__ (self,training_pipeline_config:TrainingPipeline):
        self.data_transformation_dir:str=os.path.join(
            training_pipeline_config.artifact_dir,"DataTransformation"
        )
        self.transformed_train_file_path: str = os.path.join( self.data_transformation_dir, "Transformed Data"
            "train.csv".replace("csv", "npy"),)
        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir,  "Transformed Data",
            "test.csv".replace("csv", "npy"),)
        self.transformed_val_file_path: str = os.path.join(self.data_transformation_dir, "Transformed Data",
            "val.csv".replace("csv", "npy"),)
        self.transformed_object_file_path: str = os.path.join( self.data_transformation_dir,"Transformation Object",
            "transformation_object.pkl",)

class ModelTrainer:
        def __init__ (self, training_pipeline_config:TrainingPipeline):
            self.model_trainer_dir: str = os.path.join(
                training_pipeline_config.artifact_dir, "Model Trainer")
            self.trained_model_file_path: str = os.path.join(
                self.model_trainer_dir,"Trained Model", "model.pkl")