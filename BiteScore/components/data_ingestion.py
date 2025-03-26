import os
import sys
import pymongo
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from BiteScore.Logging.logger import logger
from BiteScore.Exception.exception import BiteScoreException
from BiteScore.config.entity_config import DataIngestion
from BiteScore.config.artifact_config import DataIngestionArtifact

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

class DataIngestion:
    def __init__ (self, data_ingestion_config: DataIngestion):
        try:
            logger.info("Initializing the data ingestion configuration")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            logger.error("Error in initializing the data ingestion configuration")
            raise BiteScoreException(e, sys)
    
    def read_data_from_db(self):
        try:
            logger.info("Reading data from database")
            self.client = pymongo.MongoClient(MONGO_URI)
            db_name = self.data_ingestion_config.database_name
            train_collection = self.data_ingestion_config.collection_name_train
            test_collection = self.data_ingestion_config.collection_name_test
            val_collection = self.data_ingestion_config.collection_name_val

            df_train = pd.DataFrame(list(train_collection.find()))
            df_test = pd.DataFrame(list(test_collection.find()))
            df_val = pd.DataFrame(list(val_collection.find()))

            logger.info("Data read from database successfully")
            return df_train, df_test, df_val
        except Exception as e:
            logger.error("Error in reading data from database")
            raise BiteScoreException(e, sys)

    def save_data(self, df: pd.DataFrame, path: str):
        try:
            logger.info("Saving data...")
            df.to_csv(path, index=False)
            logger.info("Data saved locally successfully")
        except Exception as e:
            logger.error("Error in saving the data locally")
            raise BiteScoreException(e, sys)

    def initiate_ingestion(self):
        try:
            logger.info("Initiating data ingestion")
            df_train, df_test, df_val = self.read_data_from_db()
            self.save_data(df_train, self.data_ingestion_config.training_file_path)
            self.save_data(df_test, self.data_ingestion_config.testing_file_path)
            self.save_data(df_val, self.data_ingestion_config.validation_file_path)

            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
                val_file_path=self.data_ingestion_config.validation_file_path
            )
            return data_ingestion_artifact
        except Exception as e:
            logger.error("Error in initiating data ingestion")
            raise BiteScoreException(e, sys)