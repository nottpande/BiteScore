import os
import sys
import json
import certifi
from dotenv import load_dotenv

import pandas as pd
from pathlib import Path
import pymongo

from BiteScore.utils.functionalities import read_yaml
from BiteScore.Logging.logger import logger
from BiteScore.Exception.exception import BiteScoreException

MONGO_DB_URL=os.getenv("MONGO_URI")
ca=certifi.where()
load_dotenv()

class PushingDataToDB():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise BiteScoreException(e,sys)
        
    def csv_to_json_convertor(self,file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise BiteScoreException(e,sys)
        
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records

            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise BiteScoreException(e,sys)

if __name__ == "__main__":
    try:
        logger.info("Reading YAML file")
        params = read_yaml(Path("params.yaml"))
        logger.info("Pushing Data to MongoDB")
        
        DB_NAME = params.Database_Details.Database_Name
        TRAIN_COLLECTION = params.Database_Details.Train_Collection
        TEST_COLLECTION = params.Database_Details.Test_Collection
        VAL_COLLECTION = params.Database_Details.Validation_Collection
        TRAIN_FILE = params.file_locations.Formatted.Train
        TEST_FILE = params.file_locations.Formatted.Test
        VAL_FILE = params.file_locations.Formatted.Validation

        push_data = PushingDataToDB()

        logger.info("Pushing the training data to the database")
        records = push_data.csv_to_json_convertor(TRAIN_FILE)
        push_data.insert_data_mongodb(records,DB_NAME,TRAIN_COLLECTION)
        logger.info("Training data pushed to MongoDB")

        logger.info("Pushing the testing data to the database")
        records = push_data.csv_to_json_convertor(TEST_FILE)
        push_data.insert_data_mongodb(records,DB_NAME,TEST_COLLECTION)
        logger.info("Testing data pushed to MongoDB")

        logger.info("Pushing the validation data to the database")
        records = push_data.csv_to_json_convertor(VAL_FILE)
        push_data.insert_data_mongodb(records,DB_NAME,VAL_COLLECTION)
        logger.info("Validation data pushed to MongoDB")

    except Exception as e:
        logger.error("An error occurred while pushing the data to MongoDB")
        raise BiteScoreException(e,sys)

