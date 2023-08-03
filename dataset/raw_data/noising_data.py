import numpy as np 
import pandas as pd
import sys, os
from ast import literal_eval


def noising_data(df, columns, fracs):

    if len(columns) == len(fracs):
        for i in range(len(columns)):
            number_of_row = len(df)

            if fracs[i] >= 0 and fracs[i] <=1:
                amount = int(number_of_row*fracs[i])
            else:
                amount = 0

            # index list of sample
            indices = np.random.choice(range(number_of_row), size=amount, replace=False)

            # Change value
            df.loc[indices, columns[i]] = None
        return df

    else: 
        return 0


if __name__ == '__main__':
    path = r'dataset\raw_data\weather.csv'
    df = pd.read_csv(path)

    # Parameters
    columns = sys.argv[1].split() if len(sys.argv) > 1 else None
    frac = [ float(v) for v in sys.argv[2].split()] if len(sys.argv) > 2 else [0]
    value = sys.argv[3] if len(sys.argv) > 3 else None

    # Add nosing
    noise_df = noising_data(df, columns, frac)

    # Save file
    nosie_path = r'dataset\raw_data\noising_data.csv'
    df.to_csv(nosie_path, index=False)






  