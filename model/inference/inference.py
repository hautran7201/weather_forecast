import sys

sys.path.insert(3, "./")

import pandas as pd
import parameter
import utils
import preprocessor
from gru_model import GRU_model

# Load data
path = r'dataset\raw_data\weather.csv'
df = pd.read_csv(path).drop('Unnamed: 0', axis=1)

# Load paramters
parameter = utils.pickle_load(r'parameter\parameter.pickle')

# Load preprocessor
preprocessor = utils.pickle_load(r'preprocessor\preprocessor.pickle')

# Load model
model = utils.pickle_load(r'model\train\model.pkl')

# Filter data
province = 'Bac Lieu'
inference_df = df[df['province']==province][-parameter.PastLength:]

# Preprocessing
encode_columns = ["province"]  # Encode feature
scale_columns = ["max","min","rain","pressure","humidi","cloud","wind",]  # Scale feature
encoded_province = preprocessor.categorical_transform(inference_df.iloc[:1], ['province']).values.tolist()[0]
clean_data = preprocessor.get_clean_data(inference_df, scale_columns=scale_columns, encode_columns=encode_columns)
inference_data = clean_data.values.tolist()

# Inference
number_of_days = 5
result = model.inference(encoded_province, inference_data, number_of_days)
result_df = pd.DataFrame(result, columns=scale_columns)

# Rescale data
result = preprocessor.rescaling_data(result_df, scale_columns)
print(result)

