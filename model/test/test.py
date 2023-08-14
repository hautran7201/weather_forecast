import sys
sys.path.insert(2, "./")

import pandas as pd
import utils
from data_generator import *


def test(
        province,
        model_path=r'model\train\GRU_Model.pkl',
        save_plot=None
):
    # Load test data
    X_test = np.array(utils.pickle_load(r"dataset\train_val_data\X_test.pkl")[province])
    Y_test = np.array(utils.pickle_load(r"dataset\train_val_data\Y_test.pkl")[province])

    # Load model
    model = utils.pickle_load(model_path)

    # Load parameters
    parameters = utils.pickle_load(r'parameter\parameter.pickle')

    # Evaluation model
    Y, predict, loss = model.evaluation(X_test, Y_test)

    # Visualize the first predicted day
    column_name = parameters.TargetFeature
    dayth = 0
    
    Y_df = pd.DataFrame(Y[:, dayth, :], columns=column_name)
    
    predict_df = pd.DataFrame(predict[:, dayth, :], columns=column_name)

    if save_plot != None:
        utils.evaluation_plot(Y_df, predict_df, save_as=save_plot)

    return Y_df, predict_df


test('Bac Lieu')