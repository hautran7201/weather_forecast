import numpy as np
from tqdm import tqdm
from collections import defaultdict


class Generator:
    def __init__(self, data, target, distinguish_feature):
        self.data = data
        self.target = data[target+[distinguish_feature]]
        self.distin_feature = distinguish_feature

    def train_generator(
        self, past, step, fureture_length=1, ratio=None
    ):
        X_train = []
        Y_train = []
        X_val = []
        Y_val = []

        for obj in self.data[self.distin_feature].unique():
            x_province = self.data[self.data[self.distin_feature] == obj]
            y_province = self.target[self.target[self.distin_feature] == obj]

            split = int(len(x_province) * ratio)

            x_train = x_province[:split]
            x_val = x_province[split:]

            y_train = y_province[:split]
            y_val = y_province[split:]

            print(f"\n\n\nProvince {obj}")
            print(f"Training data")
            for i in tqdm(range(len(x_train) - (past + fureture_length))):
                X_train.append(
                    x_train.drop(self.distin_feature, axis=1)
                    .values[range(i, i + past, step)]
                    .tolist()
                )
                Y_train.append(
                    y_train.drop(self.distin_feature, axis=1)
                    .values[(i + past) : (i + past + fureture_length)]
                    .flatten()
                    .tolist()
                )

            print("Validation data")
            for i in tqdm(range(len(x_val) - (past + fureture_length))):
                X_val.append(
                    x_val.drop(self.distin_feature, axis=1)
                    .values[range(i, i + past, step)]
                    .tolist()
                )
                Y_val.append(
                    y_val.drop(self.distin_feature, axis=1)
                    .values[(i + past) : (i + past + fureture_length)]
                    .flatten()
                    .tolist()
                )

        return np.array(X_train), np.array(Y_train), np.array(X_val), np.array(Y_val)

    def test_generator(self, past, step, fureture_length=1, test_sample=365):
        X_test = defaultdict(list)
        Y_test = defaultdict(list)

        for obj in self.data[self.distin_feature].unique():
            x_province = self.data[self.data[self.distin_feature] == obj][-test_sample:]
            y_province = self.target[self.target[self.distin_feature] == obj][-test_sample:]

            print(f"\n\n\nProvince {obj}")
            print("Test data")
            for i in tqdm(range(len(x_province) - (past + fureture_length))):
                X_test[obj].append(
                    x_province.drop(self.distin_feature, axis=1)
                    .values[range(i, i + past, step)]
                    .tolist()
                )
                Y_test[obj].append(
                    y_province.drop(self.distin_feature, axis=1)
                    .values[(i + past) : (i + past + fureture_length)]
                    .flatten()
                    .tolist()
                )

        return X_test, Y_test
