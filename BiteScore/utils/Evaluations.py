import sys
import torch
import torch.nn as nn
from itertools import product
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from BiteScore.Logging.logger import logger
from BiteScore.Exception.exception import BiteScoreException
from BiteScore.config.artifact_config import RegressionMetrics
from BiteScore.utils.NeuralNetwork import NeuralNetwork

class Evaluations:
    @staticmethod
    def train_models(x_train, y_train, x_test, y_test, models, params, epochs):
        try:
            logger.info("Training the models, in Evaluations.py")
            logger.info(f"Models: {models}")
            logger.info(f"Parameters: {params}")
            logger.info(f"Shape of x_train: {x_train.shape}")
            logger.info(f"Shape of y_train: {y_train.shape}")
            logger.info(f"Shape of x_test: {x_test.shape}")
            logger.info(f"Shape of y_test: {y_test.shape}")

            report = {}

            for model_name, model in models.items():
                logger.info(f"Currently training {model_name}")
                param = params.get(model_name, {})
                logger.info(f"Parameters for {model_name}: {param}")
                # Handling Scikit-learn models
                if model_name != "Neural Network":
                    gs = GridSearchCV(estimator=model, param_grid=param, cv=5, error_score='raise')
                    logger.info("Fitting the model with GridSearchCV")
                    gs.fit(x_train, y_train)
                    # Print all the parameters
                    logger.info(f"Best estimator for {model_name}: {gs.best_estimator_}")
                    logger.info(f"Best score for {model_name}: {gs.best_score_}")
                    logger.info(f"Best parameters for {model_name}: {gs.best_params_}")
                    model.set_params(**gs.best_params_)
                    model.fit(x_train, y_train)

                    # Predict on train and test sets
                    y_train_pred = model.predict(x_train)
                    y_test_pred = model.predict(x_test)

                    # Calculate R² and MSE for train and test
                    train_r2 = r2_score(y_train, y_train_pred)
                    test_r2 = r2_score(y_test, y_test_pred)
                    train_mse = mean_squared_error(y_train, y_train_pred)
                    test_mse = mean_squared_error(y_test, y_test_pred)

                    # Add to report
                    report[model_name] = {
                        "train_r2": train_r2,
                        "test_r2": test_r2,
                        "train_mse": train_mse,
                        "test_mse": test_mse
                    }

                # Handling Neural Network models
                elif model_name == "Neural Network":
                    logger.info(f"Training Neural Network")
                    print(f"Training Neural Network with parameters: {param}")
                    # Generate all combinations of hyperparameters
                    param_combinations = list(product(
                        param["input_dim"],
                        param["hidden_layers"],
                        param["dropout_rate"]
                    ))

                    for input_dim, hidden_layers, dropout_rate in param_combinations:
                        logger.info(f"Training with: input_dim={input_dim}, hidden_layers={hidden_layers}, dropout_rate={dropout_rate}")

                        # Initialize Neural Network model
                        model_nn = NeuralNetwork(input_dim, hidden_layers, dropout_rate)
                        criterion = nn.MSELoss()
                        optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.001)

                        # Convert Data to Tensors (Ensure correct shape for PyTorch)
                        X_train_tensor = torch.tensor(x_train, dtype=torch.float32)
                        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
                        X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
                        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

                        # Training Loop
                        for epoch in range(epochs):
                            model_nn.train()
                            optimizer.zero_grad()
                            y_pred = model_nn(X_train_tensor)
                            loss = criterion(y_pred, y_train_tensor)
                            loss.backward()
                            optimizer.step()

                        # Predict on train and test sets
                        model_nn.eval()
                        y_train_pred = model_nn(X_train_tensor).detach().numpy()
                        y_test_pred = model_nn(X_test_tensor).detach().numpy()

                        # Calculate R² and MSE for train and test
                        train_r2 = r2_score(y_train, y_train_pred)
                        test_r2 = r2_score(y_test, y_test_pred)
                        train_mse = mean_squared_error(y_train, y_train_pred)
                        test_mse = mean_squared_error(y_test, y_test_pred)

                        # Add the performance results to the report
                        report[f"{model_name} - {input_dim}-{hidden_layers}-{dropout_rate}"] = {
                            "train_r2": train_r2,
                            "test_r2": test_r2,
                            "train_mse": train_mse,
                            "test_mse": test_mse
                        }

            # Return the final report after all models are trained
            return report

        except Exception as e:
            logger.error(f"Error Training the model {model_name}")
            raise BiteScoreException(e, sys)
        
    @staticmethod
    def get_metrics(y_pred, y_true) -> RegressionMetrics:
        regression_metrics = RegressionMetrics(
            r2_score=r2_score(y_true=y_true, y_pred=y_pred),
            loss=mean_squared_error(y_true=y_true, y_pred=y_pred)
        )
        return regression_metrics
