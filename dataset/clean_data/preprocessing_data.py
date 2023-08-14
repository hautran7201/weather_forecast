import sys

sys.path.insert(0, "./")

import pandas as pd
import utils
import os
from parameter import ModelParameters
from preprocessor import *

def preprocessing(
        scale_columns, 
        encode_columns,
        data_path = r"dataset\raw_data\weather.csv",
        preprocessor_save_path=r"preprocessor\preprocessor.pickle",
        parameter_save_path=r"parameter\parameter.pickle",
        result_save_path=r"dataset\clean_data\cleaned_data.csv"
        ):
    
    # Download data
    df = pd.read_csv(data_path)

    # Initialize Preprocessor
    preprocessor = Preprocessor()
    add_columns = [
        "province"
    ]  # Thís feature is used for generate x, t data cho each province

    # Get cleaned data
    columns_with_missing_values = preprocessor.fill_missing_value(df, method="ffill")
    cleaned_df = preprocessor.get_clean_data(
        columns_with_missing_values,
        add_columns=add_columns,
        scale_columns=scale_columns,
        encode_columns=encode_columns,
    )
    utils.pickle_dump(preprocessor_save_path, preprocessor)

    # Define training features
    if os.path.exists(parameter_save_path):
        parameters = utils.pickle_load(parameters)
    else:
        parameters = ModelParameters()
    parameters.TrainingFeature = list(preprocessor.encoded_columns.keys()) + list(preprocessor.scaled_columns.keys())
    utils.pickle_dump(parameters, parameters)

    # Save file
    cleaned_df.to_csv(result_save_path, index=False)

    return cleaned_df