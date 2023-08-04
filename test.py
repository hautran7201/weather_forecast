import pandas as pd
from model.test.test import test

feature = 'max'
province = 'Bac Lieu'
y_df, predict_df = test(province=province)

y = y_df[[feature]]
y.columns = ['True value']

predict = predict_df[[feature]]
predict.columns = ['Predict']

print(pd.concat([y, predict], axis=1))