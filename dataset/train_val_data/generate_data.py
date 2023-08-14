import sys

sys.path.insert(1, "./")

import pandas as pd
import utils
import os
from parameter import ModelParameters
from data_generator import *

def generate_data(
        dataset_path=r"dataset\clean_data\cleaned_data.csv",
        parameter_path=r"parameter\parameter.pickle",
        file_save_path=r'dataset\train_val_data'
):
    # Load data
    cleaned_df = pd.read_csv(dataset_path)

    # columns: ['province', 'province_0', 'province_1', 'province_2', 'province_3','province_4', 'province_5', 'scaled_max', 'scaled_min', 'scaled_rain','scaled_pressure', 'scaled_humidi', 'scaled_cloud', 'scaled_wind']
    selected_feature = cleaned_df.columns[-7:]

    # Parameters
    ratio = 0.8  # Split ratio
    df_length = len(cleaned_df)  # Number of sample
    step = 1  # Distance between days
    past = 60  # Number of days in the past used to predict
    future_length = 5  # Number of predicted days in the future
    distin_feature = "province"  # Prepare data for each province
    target_feature = list(selected_feature)  # Feature for output

    # Generator
    generator = Generator(cleaned_df, target_feature, distinguish_feature=distin_feature)

    # Split data
    X_train, Y_train, X_val, Y_val = generator.train_generator(past, step, fureture_length=future_length, ratio=ratio)
    X_test, Y_test = generator.test_generator(past, step, fureture_length=future_length)

    # Save
    np.save(file_save_path+r"\X_train.npy", X_train)
    np.save(file_save_path+r"\Y_train.npy", Y_train)
    np.save(file_save_path+r"\X_val.npy", X_val)
    np.save(file_save_path+r"\Y_val.npy", Y_val)
    utils.pickle_dump(file_save_path+r"\X_test.pkl", X_test)
    utils.pickle_dump(file_save_path+r"\Y_test.pkl", Y_test)

    # Save parameters
    if os.path.exists(parameter_path) == True:
        parameters = utils.pickle_load(parameters)
    else:    
        parameters = ModelParameters()
        
    parameters.Step = step  # Distance between days
    parameters.PastLength = past # Number of days in the past used to predict
    parameters.FutureLength = future_length  # Number of predicted days in the future
    parameters.DistinFeature = distin_feature # Prepare data for each province
    parameters.TargetFeature = target_feature # Feature for output
    utils.pickle_dump(parameter_path, parameters)  # Save parameters