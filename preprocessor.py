import pandas as pd
import numpy as np
from category_encoders import BinaryEncoder
from sklearn.preprocessing import MinMaxScaler


class Preprocessor:
    def __init__(self):
        # Scale feature
        self.scaler = {}
        self.scaled_columns = {}

        # Encode feature
        self.encoder = {}
        self.encoded_columns = {}

    def check_missing_value(self):
        features_with_missing = self.df.columns[self.df.isnull().any()].tolist()
        number_of_missing = self.df[features_with_missing].isnull().sum()

        missing_df = pd.DataFrame(
            {
                "column name": features_with_missing,
                "number of null": number_of_missing,
                "percent of missing": (number_of_missing * 100 / len(self.df)).round(
                    decimals=2
                ),
            }
        )
        return missing_df

    def fill_missing_value(self, df, method=""):
        df = df.fillna(method=method, inplace=False)
        return df

    def categorical_encoding(self, df, columns):
        encoded_df = pd.DataFrame()

        for column in columns:
            if column in df.columns:
                self.encoder[column] = BinaryEncoder(cols=[column])
                encoded_feature = self.encoder[column].fit_transform(df[column])
                self.encoded_columns[column] = encoded_feature.columns
                encoded_df = pd.concat([encoded_df, encoded_feature], axis=1)

        return encoded_df

    def categorical_transform(self, df, columns):
        transformed_df = pd.DataFrame()

        for column in columns:
            if column in self.encoder:
                trans_data = self.encoder[column].transform(df[column])
                transformed_df = pd.concat([transformed_df, trans_data], axis=1)
        return transformed_df

    def scaling_data(self, df, columns):
        scaled_df = pd.DataFrame()

        for column in columns:
            if column in df.columns:
                self.scaled_columns[column] = "_".join(["scaled", column])
                self.scaler[column] = MinMaxScaler()
                scaled_data = pd.DataFrame(
                    self.scaler[column].fit_transform(df[[column]]),
                    columns=[self.scaled_columns[column]]
                )
                scaled_df = pd.concat([scaled_df, scaled_data], axis=1)
        return scaled_df

    def scaling_transform(self, df, columns):
        scaled_df = pd.DataFrame()

        for column in columns:
            if column in df.columns and column in self.scaler:
                scaled_data = pd.DataFrame(
                    self.scaler[column].transform(df[column]),
                    columns=self.scaled_columns[column],
                )
                scaled_df = pd.concat([scaled_df, scaled_data], axis=1)
        return scaled_df

    def rescaling_data(self, df, columns):
        rescaled_df = pd.DataFrame()

        for column in columns:
            if column in self.scaled_columns:
                rescaled_data = pd.DataFrame(
                    np.round(self.scaler[column].inverse_transform(df[[column]]), 2),
                    columns=[column],
                )
                rescaled_df = pd.concat([rescaled_df, rescaled_data], axis=1)

        return rescaled_df

    def get_clean_data(self,df,add_columns=[],scale_columns=[],encode_columns=[]):
        df.reset_index(drop=True, inplace=True)

        if encode_columns != []:
            for column in encode_columns:
                if column not in self.encoder:
                    self.categorical_encoding(df, [column])

        add_data = df[add_columns]
        scaled_data = self.scaling_data(df, scale_columns)
        encoded_data = self.categorical_transform(df, encode_columns)
        return pd.concat([add_data,encoded_data,scaled_data], axis=1)
