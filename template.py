'''
The main use of this file template.py is to create a generic project structure.
'''

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name="BiteScore"
list_of_files = [
    ".github/workflows/.gitkeep",  # Placeholder file to keep the workflows folder in the repository for CI/CD pipeline setup using GitHub Actions.
    f"{project_name}/__init__.py",  # Makes the '{project_name}' directory a package.
    f"{project_name}/components/__init__.py",  # Initializes the 'components' module, which will hold reusable components for building pipelines.
    f"{project_name}/utils/__init__.py",  # Initializes the 'utils' module for generic utility functions used across the project.
    f"{project_name}/utils/functionalities.py",  # Contains common utility functions used in the project.
    f"{project_name}/config/__init__.py",  # Initializes the 'config' module.
    f"{project_name}/config/configuration.py",  # Manages configuration loading and validation.
    f"{project_name}/pipeline/__init__.py",  # Initializes the 'pipeline' module, which will include different data pipelines.
    f"{project_name}/entity/__init__.py",  # Initializes the 'entity' module, which defines the project's core entities (e.g., data classes).
    f"{project_name}/entity/config_entity.py",  # Defines configuration-related entities as data classes.
    f"{project_name}/constants/__init__.py",  # Initializes the 'constants' module for storing project-wide constants.
    f"{project_name}/Logging/__init__.py"  # Creating a folder to perform logging.
    f"{project_name}/Logging/logger.py" # File created for custom logging
    f"{project_name}/Exception/__init__.py" # Creating a folder to handle the exceptions.
    f"{project_name}/Exception/exception.py" # Creating a file for custom exception and its logging.
    f"{project_name}/Cloud/__init__.py" # Creating a folder to handle some details about cloud.
    "config/config.yaml",  # Stores project configuration details (e.g., paths, model parameters).
    "params.yaml",  # Stores hyperparameters and other project settings (e.g., input/output paths).
    "schema.yaml",  # Defines the schema for the dataset (e.g., column names, data types).
    "main.py",  # The main entry point of the project.
    "Dockerfile",  # Defines the instructions to build a Docker image for the project.
    "requirements.txt",  # Lists all the required Python packages to run the project.
    "setup.py",  # Makes the project installable as a package.
    "Analysis.ipynb",  # Jupyter Notebook for Exploratory Data Analysis (EDA) on the dataset.
    "templates/index.html",  # HTML template for the frontend, we will be using Flask.
]

for filepath in list_of_files:
    filepath=Path(filepath) # Creating that particular filepath
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True) # skipping if file already exists.
        logging.info(f"Creating directory {filedir} for the file: {filename}") # logging the file creation

    if(not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0): # checking if file path does not exist / file not available
        with open(filepath, 'w') as f: # Open the path, and create the file
            pass # No content to put, hence pass.
            logging.info(f"Creating Empty File : {filepath}")
    else:
        logging.info(f"File {filename} already exists!")
