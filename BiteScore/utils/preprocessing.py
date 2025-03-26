'''
This file contains functions to perform the preprocessing transformations in our dataset.
'''

import re
import os
from pathlib import Path
from ensure import ensure_annotations
from BiteScore.utils.functionalities import read_yaml, load_json
from BiteScore.Logging.logger import logger

@ensure_annotations
def label_mapping(label:str) -> int:
    '''
    This function is made to map:
        'Yes' -> 1
        'No' -> 0
    '''
    return label.map({'Yes':1, 'No':0})

@ensure_annotations
def extract_rating(rating:str) -> float:
    ''' 
    This function takes the rating in the dataset (which is a string) and it converts the rating to a float number
    '''
    # Check if the rating matches the format digit.digit/5
    if re.match(r'^\d+\.\d+/5$', str(rating)):
        logger.info("Extracted rating from string")
        return float(rating.split('/')[0])
    else:
        return None

@ensure_annotations
def price_per_head(price: str) -> float:
    ''' 
    This function takes the approx cost for two people in the dataset (which is a string), 
    converts it to a float number, then divides by 2 to get the cost per head.
    '''
    if ',' in price:
        price = price.replace(',', '')  # Reassign the modified string to price
        logger.info("Got the price per head")
    return float(price) / 2  # Convert the cleaned price to float and divide by 2

@ensure_annotations
def get_mappings(type:str) -> dict:
    '''
    This function returns the mapping required for mapping
    '''
    params_file = os.path.join(os.getcwd(),"params.yaml")
    print(params_file)
    yaml_data = read_yaml(Path(params_file))

    if type=="Resaurant Type":
        # load the restaurant type mapping
        file_location = yaml_data['preprocess']['mapping']['restaurant_type']
        mapping = load_json(Path(file_location))
        logger.info("Loaded the Restaurant Type Mapping Successfully!")
        return mapping
    
    elif type=="Cuisine":
        # load the cuisine mapping
        file_location = yaml_data['preprocess']['mapping']['cuisine']
        mapping = load_json(Path(file_location))
        logger.info("Loaded the Cuisine Mapping Successfully!")
        return mapping

    elif type=="Location":
        # load the location mapping
        file_location = yaml_data['preprocess']['mapping']['location']
        mapping = load_json(Path(file_location))
        logger.info("Loaded the Location Mapping Successfully!")
        return mapping
    
    else:
        # The option given is incorrect
        logger.error(f"{type} mapping does not exist.")

@ensure_annotations
def perform_mapping(entities_list:list, type:str) -> list: 
    items = entities_list.split(', ')
    mapped_items = []
    mapping = get_mappings(type)
    for key, values in mapping.items():
        if any(item in items for item in values):
            mapped_items.append(key)
    logger.info(f"Completed mapping for {type} list given")
    return ', '.join(mapped_items) if mapped_items else 'Miscellaneous'

@ensure_annotations
def encode_location(location:str) -> int:
    location_mapping = get_mappings(type="Location")
    if location in location_mapping:
        logger.info(f"Encoded Location")
        return location_mapping[location]
    else:
        logger.info(f"Encoded Location")
        return len(location_mapping) + 1

@ensure_annotations
def score_reviews(reviews: list, sentiment_scoring_model) -> float:
    if not reviews:  # Avoid division by zero
        logger.warning("No reviews to score.")
        return 0.0
    
    scores = []
    for tuple in reviews:
        rating = tuple[0]
        review = tuple[1]
        result = sentiment_scoring_model(review)
        result = result[0]  # Getting the dictionary of the result

        label = result['label']
        confidence = result['score']

        if label == 'POSITIVE':
            score = (rating / 5) * 1 * (0.1 * confidence)  # 10% confidence weight
        elif label == 'NEGATIVE':
            score = (rating / 5) * -1 * (0.1 * confidence)  # 10% confidence weight
        else:
            score = 0  # If an unexpected label is encountered
        
        scores.append(score)

    final_score = sum(scores) / len(scores) if scores else 0
    logger.info(f"Scored the reviews")
    return final_score