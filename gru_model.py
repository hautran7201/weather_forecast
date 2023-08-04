import numpy as np
import pandas as pd
import tensorflow as tf

import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping


class GRU_model:
    def __init__(
        self,
        input_shape,
        output_shape,
        past_length,
        future_length,
        unit_per_layer=[],
        loss="mean_squared_error",
        optimizer="adam",
        model_path=None,
    ):
        # Parameters
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.past_length = past_length
        self.future_length = future_length

        # Model
        if model_path == None:
            self.model = Sequential()
            self.model.add(
                GRU(
                    unit_per_layer[0],
                    return_sequences=True,
                    input_shape=self.input_shape,
                )
            )

            for unit in unit_per_layer[1:-1]:
                self.model.add(GRU(unit, return_sequences=True))

            self.model.add(GRU(unit_per_layer[-1]))
            self.model.add(Dense(self.output_shape))
            self.model.compile(loss=loss, optimizer=optimizer)
        else:
            self.model = tf.keras.models.load_model(model_path)

    def train(self, train, val, batch_size, epochs, path=None, patience=None):
        history = []

        early_stopping = EarlyStopping(monitor="val_loss", patience=patience)

        history = self.model.fit(
            train[0],
            train[1],
            validation_data=(val[0], val[1]),
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            verbose=1,
            callbacks=[early_stopping],
        )

        if path != None:
            self.model.save(path)

        return self.model, history

    def evaluation(self, X, Y):
        num_of_feature = int(self.output_shape[0] / self.future_length)

        predict = self.model.predict(X).reshape(1, -1)[0]
        predict = np.array(
            [
                predict[i : i + self.output_shape[0]].reshape((-1, num_of_feature))
                for i in range(0, len(predict), self.output_shape[0])
            ]
        )
        Y = np.array(Y).reshape((-1, self.future_length, num_of_feature))

        losses = {}

        for dayth in range(self.future_length):
            prediction = pd.DataFrame(predict[:, dayth, :])
            true_value = pd.DataFrame(Y[:, dayth, :])
            losses["day " + str(dayth)] = utils.mean_square_error(
                prediction, true_value
            )
        return Y, predict, losses

    def inference(self, encoded_province, df, number_of_days):
        dayth = 0
        predict_days = []
        print(self.past_length)
        for day in range(number_of_days):
            predicted_day = (
                self.model.predict([df[day : (day + self.past_length)]])[0]
                .reshape((self.future_length, int(self.output_shape[0]/self.future_length)))[dayth]
                .tolist()
            )
            df.append(encoded_province + predicted_day)
            predict_days.append(predicted_day)

        return predict_days
