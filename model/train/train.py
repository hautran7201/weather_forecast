import sys

sys.path.insert(4, './')

import utils
import numpy as np
from parameter import ModelParameters
from gru_model import GRU_model

def train(
        data_folder=r"dataset\train_val_data",
        batch_size = 512,
        epochs = 100,
        model_path = r"model\train\model.h5",
        patience_of_early_stopping = 3,
        ):
    # Load data
    X_train = np.load(data_folder + "\\" + "X_train.npy")
    Y_train = np.load(data_folder + "\\" + "Y_train.npy")
    X_val = np.load(data_folder + "\\" + "X_val.npy")
    Y_val = np.load(data_folder + "\\" + "Y_val.npy")

    # Load parameter
    parameters = utils.pickle_load(r'parameter\parameter.pickle')

    # Define parameters
    input_shape = parameters.InputShape
    output_shape = parameters.OutputShape
    past_length = parameters.PastLength
    future_length = parameters.FutureLength

    # Train model
    model = GRU_model(
        input_shape=input_shape, 
        output_shape=output_shape, 
        past_length=past_length,
        future_length=future_length,
        model_path=model_path
    )

    """model.train(
        (X_train, Y_train),
        (X_val, Y_val),
        batch_size=batch_size,
        epochs=epochs,
        path=model_path,
        patience=patience_of_early_stopping,
    )"""

    # Save GRU_model
    path = r'model\train\GRU_Model.pkl'
    utils.pickle_dump(path, model)

