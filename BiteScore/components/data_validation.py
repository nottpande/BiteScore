import os
import sys
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp
from BiteScore.Logging.logger import logger
from BiteScore.Exception.exception import BiteScoreException
from BiteScore.config.entity_config import DataValidation
from BiteScore.config.artifact_config import DataIngestionArtifact, DataValidationArtifact
from BiteScore.utils.functionalities import read_yaml, write_yaml_file

class DataValidation:
    def __init__ (self, data_validation_config:DataValidation, data_ingestion_artifact:DataIngestionArtifact):
        try:
            logger.info("Initializing the data validation configuration")
            self.data_validation_config=data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self._schema = read_yaml(Path("../schema.yaml"))
        except Exception as e:
            logger.error("Error in initializing the data validation configuration")
            raise BiteScoreException(e, sys)

    def read_data(data) -> pd.DataFrame:
        try:
            return pd.read_csv(data)
        except Exception as e:
            logger.error("Error in reading the data")
            raise BiteScoreException(e, sys)
    
    def validate_data(self, df) -> bool:
        try:
            logger.info("Validating the number of columns")
            num_cols = len(self._schema.columns)
            logger.info(f"Number of columns in the schema: {num_cols}")
            if num_cols == len(df.columns):
                logger.info("Data is valid")
                return True
            logger.error("Data is invalid")
            return False
        except Exception as e:
            logger.error("Error in validating the data")
            raise BiteScoreException(e, sys)
    
    def detect_drift(self, train_df, test_df, val_df, threshold=0.05) -> bool:
        try:
            status = True
            report = {}
            for column in train_df.columns:
                train_data = train_df[column]
                test_data = test_df[column]
                val_data = val_df[column]

                # Compare Train vs Test
                test_stat = ks_2samp(train_data, test_data)
                test_drift = test_stat.pvalue < threshold

                # Compare Train vs Validation
                val_stat = ks_2samp(train_data, val_data)
                val_drift = val_stat.pvalue < threshold

                # If drift is found in either comparison, set status to False
                if test_drift or val_drift:
                    status = False

                # Update report
                report[column] = {
                    "train_vs_test": {
                        "p_value": float(test_stat.pvalue),
                        "drift_status": test_drift
                    },
                    "train_vs_val": {
                        "p_value": float(val_stat.pvalue),
                        "drift_status": val_drift
                    }
                }

            # Writing the report in our YAML file.
            os.makedirs(os.path.dirname(self.data_validation_config.drift_report_file_path),exist_ok=True)
            write_yaml_file(self.data_validation_config.drift_report_file_path, report)

        except Exception as e:
            logger.error("Error in detecting drift")
            raise BiteScoreException(e, sys)

    def initiate_validation(self) -> DataValidationArtifact:
        try:
            logger.info("Initiating data validation")
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            val_df = self.read_data(self.data_ingestion_artifact.validation_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            status = self.validate_data(train_df)
            if not status:
                logger.info("Training data is not valid, it does not contain all the columns")
            status = self.validate_data(val_df)
            if not status:
                logger.info("Validation data is not valid, it does not contain all the columns")
            status = self.validate_data(test_df)
            if not status:
                logger.info("Testing data is not valid, it does not contain all the columns")
            
            # Checking the drift between the datasets
            logger.info("Detecting drift between the datasets")
            self.detect_drift(train_df, test_df, val_df)

            # Saving the valid data
            valid_data_dir = self.data_validation_config.valid_data_dir
            os.makedirs(valid_data_dir, exist_ok=True)
            valid_train_file_path = self.data_validation_config.valid_train_file_path
            valid_val_file_path = self.data_validation_config.valid_val_file_path
            valid_test_file_path = self.data_validation_config.valid_test_file_path

            train_df.to_csv(valid_train_file_path, index=False, header=True)
            val_df.to_csv(valid_val_file_path, index=False, header=True)
            test_df.to_csv(valid_test_file_path, index=False, header=True)

            # Creating our Data Validation Artifact
            logger.info("Creating Data Validation Artifact")
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=valid_train_file_path,
                valid_val_file_path=valid_val_file_path,
                valid_test_file_path=valid_test_file_path,
                invalid_train_file_path=None,
                invalid_val_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifact
        except Exception as e:
            logger.error("Error in initiating data validation")
            raise BiteScoreException(e, sys)