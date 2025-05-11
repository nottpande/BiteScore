import os
import sys
import dagshub
import mlflow
from dotenv import load_dotenv
from pathlib import Path

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from BiteScore.Logging.logger import logger
from BiteScore.Exception.exception import BiteScoreException
from BiteScore.config.artifact_config import DataTransformationArtifact, ModelTrainerArtifact
from BiteScore.config.entity_config import ModelTrainer
from BiteScore.utils.Evaluations import Evaluations
from BiteScore.utils.functionalities import save_model, load_model, load_numpy_array_data
load_dotenv()

class BiteScoreModel:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

class ModelTrainer:
    def __init__ (self, model_trainer_config: ModelTrainer, data_transformation_Artifact: DataTransformationArtifact):
        try:
            logger.info("Initializing the model trainer configuration")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_Artifact = data_transformation_Artifact
        except Exception as e:
            logger.error("Error in initializing the model trainer configuration")
            raise BiteScoreException(e, sys)
    
    def track(self, best_model, metrics):
        dagshub.init(repo_owner='nottpande', repo_name='BiteScore', mlflow=True)

        with mlflow.start_run():
            # tracking the metric
            mlflow.log_metric("R2 Score", metrics.r2_score)
            mlflow.log_metric("Loss", metrics.loss)

            # tracking the model
            model_type = type(best_model).__module__
            if "sklearn" in model_type:
                mlflow.sklearn.log_model(best_model, "Model")
            elif "torch" in model_type:
                mlflow.pytorch.log_model(best_model, "Model")
    
    def train_models(self, x_train, y_train, x_test, y_test, x_val, y_val):
        try:
            logger.info("Training the models")
            models = {
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "Polynomial Regression": Pipeline([
                    ("Polynomial Features", PolynomialFeatures(degree=2)),
                    ("Linear Regression", LinearRegression())
                ]),
                "Neural Network": "pytorch_custom" 
            }

            params = {
                "Decision Tree Regressor": {
                    'criterion': ['squared_error', 'friedman_mse']
                },
                "Linear Regression": {
                    'fit_intercept': [True, False]
                },
                "Polynomial Regression": {
                    'Polynomial Features__degree': [2, 3, 4]
                },
                "Neural Network": {
                    'input_dim': [x_train.shape[1]], 
                    'hidden_layers': [[32, 16, 8], [64, 32, 16], [128, 64, 32, 16, 8]],
                    'dropout_rate': [0.1, 0.2, 0.3]
                }
            }

            report = Evaluations.train_models(x_train, y_train, x_test, y_test, models, params, epochs=25)
            logger.info(f"Model training report: {report}")
            # Getting the best model based on the report
            best_model_score = max(report, key=lambda model: report[model]["test_r2"])
            best_model = models[best_model_score]
            logger.info(f"Best model: {best_model_score} with score: {report[best_model_score]['test_r2']}")

            # Getting Metrics based on the best model
            y_train_pred=best_model.predict(x_train)
            train_metrics = Evaluations.get_metrics(y_train, y_train_pred)
            logger.info(f"Train metrics: {train_metrics}")
            # Tracking the best model
            self.track(best_model, train_metrics)

            # Same for testing
            y_test_pred=best_model.predict(x_test)
            test_metrics = Evaluations.get_metrics(y_test, y_test_pred)
            self.track(best_model, test_metrics)

            # Same for validation
            y_val_pred=best_model.predict(x_val)
            val_metrics = Evaluations.get_metrics(y_val, y_val_pred)
            self.track(best_model, val_metrics)

            preprocessor = load_model(Path(self.data_transformation_Artifact.transformed_object_file_path))
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)

            model_instance = BiteScoreModel(preprocessor, best_model)
            save_model(model_instance, Path(self.model_trainer_config.trained_model_file_path))
            save_model(path = Path("final_model/model.pkl"), data = best_model)

            # Creating the model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metrics,
                test_metric_artifact=test_metrics,
                val_metric_artifact=val_metrics
            )
            logger.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            logger.error("Error in training the models")
            raise BiteScoreException(e, sys)

    def initiate_trainer(self):
        try:
            logger.info("Initiating the model trainer")
            train_array = load_numpy_array_data(file_path=self.data_transformation_Artifact.transformed_train_file_path)
            test_array = load_numpy_array_data(file_path=self.data_transformation_Artifact.transformed_test_file_path)
            val_array = load_numpy_array_data(file_path=self.data_transformation_Artifact.transformed_val_file_path)

            x_train, y_train = train_array[:,:-1], train_array[:,-1]
            x_test, y_test = test_array[:,:-1], test_array[:,-1]
            x_val, y_val = val_array[:,:-1], val_array[:,-1]

            model_trainer_artifact = self.train_models(x_train, y_train, x_test, y_test, x_val, y_val)
            return model_trainer_artifact
        except Exception as e:
            logger.error("Error in initiating the model trainer")    
            raise BiteScoreException(e, sys)
