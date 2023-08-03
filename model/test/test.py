import sys

sys.path.insert(2, "./")

import pandas as pd
import parameter as pt
import utils
from gru_model import GRU_model
from data_generator import *

# Load parameters
paramaters = utils.pickle_load(r'parameter\parameter.pickle')

# Choice province for testing
province = "Bac Lieu"

# Load test data
X_test = np.array(utils.pickle_load(r"dataset\train_val_data\X_test.pkl")[province])
Y_test = np.array(utils.pickle_load(r"dataset\train_val_data\Y_test.pkl")[province])

# Load model
model = utils.pickle_load(r'model\train\model.pkl')

# Evaluation model
Y, predict, loss = model.evaluation(X_test, Y_test, future_length=paramaters.FutureLength)

# Visualize the first predicted day
column_name = [
    "scaled_max",
    "scaled_min",
    "scaled_rain",
    "scaled_pressure",
    "scaled_humidi",
    "scaled_cloud",
    "scaled_wind",
]
dayth = 0
Y_df = pd.DataFrame(Y[:, dayth, :], columns=column_name)
predict_df = pd.DataFrame(predict[:, dayth, :], columns=column_name)

path = rf"model\test\image\{province}_{dayth}.png"
utils.evaluation_plot(Y_df, predict_df, save_as=path)
