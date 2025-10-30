#all the common things which we will use in entire project like reading and writing data,
# training the model,evaluation etc

import os
import sys
import pandas as pd
import numpy as np
import dill
from src.exception import CustomException
from src.logger import logging

def save_object(file_path: str, obj: object) -> None:
    """Saves a Python object to a file using pickle."""
    try: 
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)