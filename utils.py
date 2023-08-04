import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf


def mean_square_error(A, B):
    mse = np.mean((A - B) ** 2)
    return mse


def pickle_dump(path, X_test):
    with open(path, "wb") as file:
        pickle.dump(X_test, file)


def pickle_load(path):
    with open(path, "rb") as file:
        return pickle.load(file)
    

def tf_load(path):
    return tf.keras.models.load_model(path)


def evaluation_plot(true_value, prediction, show=True, save_as=None):
    feature_name = true_value.columns

    plt.figure(figsize=(22, 22))
    plt.subplots_adjust(wspace=0.4, hspace=0.8)

    for i in range(len(feature_name)):
        plt.subplot(len(feature_name), 1, i + 1)
        plt.plot(
            range(len(prediction)), prediction[feature_name[i]], label="Predict value"
        )
        plt.plot(
            range(len(true_value)), true_value[feature_name[i]], label="True value"
        )

        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.title((feature_name[i]), fontsize=10)
        plt.legend(fontsize=8)

    if save_as != None:
        plt.savefig(save_as)

    if show:
        plt.show()