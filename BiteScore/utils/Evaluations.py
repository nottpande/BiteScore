import sys
import torch
import torch.nn as nn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from itertools import product
from BiteScore.Logging.logger import logger
from BiteScore.Exception.exception import BiteScoreException
from BiteScore.config.artifact_config import RegressionMetrics
from BiteScore.utils.NeuralNetwork import NeuralNetwork

class Evaluations:
    @staticmethod
    def train_models(x_train, y_train, x_test, y_test, models, params, epochs):
        try:
            logger.info("Training the models")
            report = {}

            for model_name, model in models.items():
                param = params.get(model_name, {})

                # Handling Scikit-learn models
                if model_name != "Neural Network":
                    logger.info(f"Currently training {model_name}")
                    gs = GridSearchCV(model, param, cv=5)
                    gs.fit(x_train, y_train)
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

                    param_combinations = list(product(
                        param["Neural Network"]["input_dim"],
                        param["Neural Network"]["hidden_layers"],
                        param["Neural Network"]["dropout_rate"]
                    ))

                    for input_dim, hidden_layers, dropout_rate in param_combinations:
                        model_nn = NeuralNetwork(input_dim, hidden_layers, dropout_rate)
                        criterion = nn.MSELoss()
                        optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.001)

                        # Convert Data to Tensors
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

                        # Add to report
                        report[f"{model_name} - {input_dim}-{hidden_layers}-{dropout_rate}"] = {
                            "train_r2": train_r2,
                            "test_r2": test_r2,
                            "train_mse": train_mse,
                            "test_mse": test_mse
                        }

            # Return the final report
            return report

        except Exception as e:
            logger.error(f"Error Training the model {model_name}")
            raise BiteScoreException(e, sys)
        
    @staticmethod
    def get_metrics(y_pred, y_true) -> RegressionMetrics:
        regression_metrics = RegressionMetrics(
            r2_score= r2_score(y_true=y_true, y_pred=y_pred),
            loss= mean_squared_error(y_true=y_true, y_pred=y_pred)
        )
        return regression_metrics