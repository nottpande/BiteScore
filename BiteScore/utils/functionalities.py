'''
This file contains some of the common functionalities that will be used in the project.
'''

import os
import yaml
import json
import joblib
import numpy as np
from BiteScore.Logging.logger import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from box.exceptions import BoxValueError

@ensure_annotations #this is to strictly follow the parameter type.
def read_yaml(yaml_path : Path) -> ConfigBox:
    '''
    This function reads the yaml file and returns config box type
    
    Args:
        yaml_path (str) : Path to the YAML file.
    Raises:
        ValueError : if YAML file is empty
        e : empty file.
    Returns:
        ConfigBox : ConfigBox Type
    '''
    try:
        with open (yaml_path) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML File : {yaml_path} loaded successfully!")
            # Sending it as ConfigBox for easy access of Key-Value pairs.
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML File is empty!")
    except Exception as e:
        raise e
    

@ensure_annotations
def create_directories(path_to_directories : list, verbose=True):
    '''
    This function is simply used to create the directories, that we need.

    Args:
        path_to_directories (list) : A list containing all the different paths.
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    '''
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """
    This function is used to save the json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"json file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    This function is used to load json files data

    Args:
        path (Path): path to json file
    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_model(data, path: Path):
    """
    This function is used to save the models.

    Args:
        data (Any): data to be saved as binary
        path (Path): path to model
    """
    # Create the directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(value=data, filename=path)
    logger.info(f"Model saved at: {path}")

@ensure_annotations
def load_model(path: Path):
    """
    This function is used to load our model

    Args:
        path (Path): path to model
    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"Model loaded from: {path}")
    return data

@ensure_annotations
def write_yaml_file(file_path: str, content: dict, replace: bool = False):
    logger.info(f"Writing to yaml file: {file_path}")
    if not isinstance(content, dict):
        raise ValueError("Content must be a dictionary.")
    if replace:
        if os.path.exists(file_path):
            os.remove(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as file:
        logger.info("Writing content to yaml file")
        yaml.dump(content, file, default_flow_style=False)
        logger.info(f"Content writing completed")

@ensure_annotations
def save_numpy_array_data(file_path: str, array: np.ndarray):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "wb") as file_obj:
        np.save(file_obj, array)

@ensure_annotations
def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    with open(file_path, "rb") as file_obj:
        return np.load(file_obj, allow_pickle=True)