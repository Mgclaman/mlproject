import sys
import os 
from dataclasses import dataclass

import numpy as np # for numerical computations
import pandas as pd # for data manipulation
from sklearn.compose import ColumnTransformer # for applying different transformations to different columns
from sklearn.impute import SimpleImputer # for handling missing values
from sklearn.pipeline import Pipeline # for creating machine learning pipelines
from sklearn.preprocessing import OneHotEncoder, StandardScaler # for categorical encoding and
#feature scaling
from src.exception import CustomException # custom exception handling
from src.logger import logging # logging setup
from src.utils import save_object # utility function to save objects

logging.info("Data Transformation initiated")
# The @dataclass decorator is used to automatically 
# add special "dunder" (double underscore) methods to your class
# __init__ (initializer), __repr__ (string representation), 
# and __eq__ (equality) methods for you.
@dataclass
class DataTransformationConfig: # it  will have path where we want to save the preprocessor object
    preprocessor_obj_file_path= os.path.join('artifacts', 'preprocessor.pkl') # path to save preprocessor object
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    def get_data_transformer_object(self):
        #This function is responsible for data transformation
        try:
            numerical_columns= ['writing_score', 'reading_score']
            categorical_columns= [
                'gender',
                'race_ethnicity', 
                'parental_level_of_education', 
                'lunch',
                'test_preparation_course'
            ]
            # Numerical Pipeline
            num_pipeline= Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            #categorical pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]   
            )
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            preprocessor= ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            logging.error(f"Error occurred while creating preprocessor: {e}")
            raise CustomException(e, sys) 
        
        
    def initiate_data_transformation(self, train_path,test_path):
        try:
                train_df= pd.read_csv(train_path)
                test_df= pd.read_csv(test_path)
                logging.info("Read train and test data completed")
                logging.info("Obtaining preprocessor object")
                preprocessor_obj= self.get_data_transformer_object()
                
                target_column_name= 'math_score'

                numerical_columns= ['writing_score', 'reading_score']
                input_feature_train_df= train_df.drop(columns=[target_column_name], axis=1)
                target_feature_train_df= train_df[target_column_name]
                
                input_feature_test_df= test_df.drop(columns=[target_column_name], axis=1)
                target_feature_test_df= test_df[target_column_name]     
                
                logging.info("Applying preprocessing object on training and testing dataframes")
                input_feature_train_arr= preprocessor_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr= preprocessor_obj.transform(input_feature_test_df)

                train_arr= np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
                test_arr= np.c_[input_feature_test_arr, np.array(target_feature_test_df)]  
                
                logging.info("Saved preprocessing object")
                save_object(
                    file_path= self.data_transformation_config.preprocessor_obj_file_path,
                    obj= preprocessor_obj
                )
                return (
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path,
                ) 

        except Exception as e:
                logging.error(f"Error occurred during data transformation: {e}")
                raise CustomException(e, sys)