import sys
from BiteScore.Logging.logger import logger
from BiteScore.Exception.exception import BiteScoreException
from BiteScore.components.data_ingestion import DataIngestion
from BiteScore.components.data_validation import DataValidation
from BiteScore.components.data_transformation import DataTransformation
from BiteScore.components.model_trainer import ModelTrainer
from BiteScore.config.entity_config import TrainingPipeline, DataIngestion as DataIngestionConfig, DataTransformation as DataTransformationConfig, DataValidation as DataValidationConfig, ModelTrainer as ModelTrainerConfig

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipeline()

        logger.info("Initiating Data Ingestion")
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_ingestion()
        logger.info("Data Ingestion Completed")

        logger.info("Initiating Data Validation")
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
        data_validation_artifact = data_validation.initiate_validation()
        logger.info("Data Validation Completed")

        logger.info("Initiating Data Transformation")
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact, data_transformation_config=data_transformation_config)
        data_transformation_artifact = data_transformation.transform_data()
        logger.info("Data Transformation Completed")

        logger.info("Initiating Model Trainer")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_Artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_trainer()
        logger.info("Model Trainer task completed")

    except Exception as e:
        logger.error("An error occurred while running main.py")
        raise BiteScoreException(e, sys)