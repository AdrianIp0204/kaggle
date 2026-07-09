import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

data = pd.read_csv("train.csv")
data = data.dropna(subset='SalePrice')

test = pd.read_csv("test.csv")

y = data['SalePrice']
X = data.drop(['SalePrice','Id'], axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42)

ordinal_orders = {
    "PoolQC": ['Na', 'Fa', 'TA', 'Gd', 'Ex'],
    "GarageCond": ['Na', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    "GarageQual": ['Na', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    "FireplaceQu": ['Na', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    "KitchenQual": ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    "HeatingQC": ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    "BsmtCond": ['Na', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    "BsmtQual": ['Na', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    "ExterQual": ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    "ExterCond": ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
}


num_col, cat_col, ord_col = [], [], []
for col, dtype in X.dtypes.items():
    if dtype == 'int64' or dtype == 'float64':
        num_col.append(col)
    elif col in ordinal_orders:
        ord_col.append(col)
    else:
        cat_col.append(col)

ordinal_categories = [ordinal_orders[col] for col in ord_col]

num_pipe = Pipeline([
    ('impute', SimpleImputer())
])

cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('OHE', OneHotEncoder(handle_unknown='ignore'))
])

ord_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ord_encode', OrdinalEncoder(categories=ordinal_categories,handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipe, num_col),
        ('cat', cat_pipe, cat_col),
        ('ord', ord_pipe, ord_col)
    ]
)

model = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

param_grid = {
    'model__n_estimators' : [350, 400, 450],
    'model__max_depth' : [23, 25, 27],
    'model__min_samples_leaf': [2]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=10,
    n_jobs=-1,
    refit=True,
    verbose=2
)

grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV MAE:", -grid.best_score_)

best_model = grid.best_estimator_

pred = best_model.predict(X_val)
score = mean_absolute_error(y_val, pred)

print("Validation MAE:", score)

grid.fit(X, y)

test_X = test.drop('Id', axis=1)

pred = grid.best_estimator_.predict(test_X)

submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': pred
    })

submission.to_csv("submission.csv", index=False)
