import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

data = pd.read_csv("train.csv")
data = data.dropna(subset='SalePrice')

test = pd.read_csv("test.csv")

y = data['SalePrice']
X = data.drop(['SalePrice','Id'], axis=1)

num_col, cat_col = [], []
for col, dtype in X.dtypes.items():
    if dtype == 'int64' or dtype == 'float64':
        num_col.append(col)
    else:
        cat_col.append(col)

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

num_pipe = Pipeline([
    ('impute', SimpleImputer()),
])

cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('OHE', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipe, num_col),
        ('cat', cat_pipe, cat_col)
    ]
)

model = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

model.fit(X, y)

test_X = test.drop('Id', axis=1)

pred = model.predict(test)

submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': pred
    })

submission.to_csv("submission.csv", index=False)
