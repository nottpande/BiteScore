import sys
import ast
import pandas as pd
from transformers import pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from BiteScore.Logging.logger import logger
from BiteScore.Exception.exception import BiteScoreException
from BiteScore.utils.preprocessing import label_mapping, encode_location, perform_mapping, score_reviews

# Custom Transformer for Binary Encoding (Yes/No columns like 'online_order' and 'book_table')
class BinaryEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_encode):
        self.columns_to_encode = columns_to_encode

    def fit(self, X, y=None):
        return self  # No fitting needed for binary encoding

    def transform(self, X):
        X_copy = X.copy()
        for column in self.columns_to_encode:
            X_copy[column] = X_copy[column].apply(lambda x: label_mapping(x))
        return X_copy

# Custom Transformer for Label Encoding (Location column with many unique locations)
class CustomEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_to_encode):
        """
        column_to_encode: The column to apply custom encoding to.
        encode_function: The custom encoding function to use (e.g., encode_location).
        """
        self.column_to_encode = column_to_encode
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.column_to_encode] = X_copy[self.column_to_encode].apply(lambda x : encode_location(x))
        
        return X_copy
class GroupingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column: str, mapping_type: str):
        """
        column: The column name in the DataFrame to apply mapping to.
        mapping_type: The mapping type to be used.
        """
        self.column = column
        self.mapping_type = mapping_type
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        # Apply the perform_mapping function to the specified column with the appropriate mapping type
        X_copy[self.column] = X_copy[self.column].apply(lambda x: perform_mapping(x, self.mapping_type))
        
        return X_copy

# Custom Transformer for Multi-Label Binarization (cuisines and restaurant types)
class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, binarizer):
        self.column_name = column_name
        self.binarizer = binarizer

    def fit(self, X, y=None):
        # Fit the MultiLabelBinarizer on the column in the data
        self.binarizer.fit(X[self.column_name])
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        # Perform the transformation
        transformed = self.binarizer.transform(X_copy[self.column_name])
        
        # Convert the list of binary values into DataFrame columns (one for each class)
        binarized_df = pd.DataFrame(transformed, columns=self.binarizer.classes_)
        
        # Drop the original column and merge the new binarized columns
        X_copy = X_copy.drop(columns=[self.column_name])
        X_copy = pd.concat([X_copy, binarized_df], axis=1)
        
        return X_copy


# Custom Transformer for Review Scoring
class ReviewScoringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sentiment_scoring_model):
        self.sentiment_scoring_model = sentiment_scoring_model

    def fit(self, X, y=None):
        return self  # No fitting needed for review scoring

    def transform(self, X):
        X_copy = X.copy()
        X_copy['reviews_score'] = X_copy['reviews'].apply(lambda x: score_reviews(ast.literal_eval(x), self.sentiment_scoring_model))
        # Drop the 'reviews' column after scoring
        X_copy.drop(columns=['reviews'], inplace=True)
        return X_copy

# Main Preprocessing Pipeline Class
class PreprocessingPipeline:
    def __init__(self, CONFIG_DATA, train_data):
        try:
            logger.info("Initializing the preprocessing pipeline configuration")
            self.config = CONFIG_DATA
            self.sentiment_scoring_model = pipeline("sentiment-analysis")
            
            # Initialize MultiLabelBinarizers
            self.multi_label_binarizer_cuisines = MultiLabelBinarizer()
            self.multi_label_binarizer_rest_types = MultiLabelBinarizer()

            # Fit the binarizers on the entire training data
            self.multi_label_binarizer_cuisines.fit(train_data['cuisines'])
            self.multi_label_binarizer_rest_types.fit(train_data['rest_type'])
        except Exception as e:
            logger.error("Error in initializing the preprocessing pipeline configuration")
            raise BiteScoreException(e, sys)

    def get_pipeline(self):
        # Return a Pipeline object for the preprocessing steps
        try:
            # Apply binary encoding for 'online_order' and 'book_table'
            binary_encoder = BinaryEncoderTransformer(columns_to_encode=['online_order', 'book_table'])

            # Apply label encoding for 'location'
            label_encoder_location = CustomEncoderTransformer(column_to_encode='location')

            # Perform Mapping for the 'cuisines' and 'rest_type' columns
            # Create the transformers for each column with its corresponding mapping type
            logger.info(f"Mapping the restaurant type")
            rest_type_transformer = GroupingTransformer(column='rest_type', mapping_type='Resaurant Type')
            logger.info(f"Performing Multi Label Binarization for restaurant")
            binarizer_rest_type = MultiLabelBinarizerTransformer(column_name='rest_type', binarizer=self.multi_label_binarizer_rest_types)

            logger.info(f"Mapping the cuisines")
            cuisines_transformer = GroupingTransformer(column='cuisines', mapping_type='Cuisine')
            logger.info(f"Performing Multi Label Binarization for cuisines")
            binarizer_cuisines = MultiLabelBinarizerTransformer(column_name='cuisines', binarizer=self.multi_label_binarizer_cuisines)

            # Review scoring step
            logger.info(f"Scoring the reviews")
            review_scoring = ReviewScoringTransformer(self.sentiment_scoring_model)

            # Create the preprocessing pipeline
            pipeline_steps = [
                ("binary_encoder", binary_encoder),
                ("label_encoder_location", label_encoder_location),
                ('rest_type_grouping', rest_type_transformer),
                ('cuisines_grouping', cuisines_transformer),
                ("binarizer_cuisines", binarizer_cuisines),
                ("binarizer_rest_type", binarizer_rest_type),
                ("review_scoring", review_scoring)
            ]

            preprocessing_pipeline = Pipeline(pipeline_steps)
            logger.info("Preprocessing pipeline created successfully")
            return preprocessing_pipeline

        except Exception as e:
            logger.error("Error in creating the preprocessing pipeline")
            raise BiteScoreException(e, sys)