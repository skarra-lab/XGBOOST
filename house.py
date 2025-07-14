import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

#Load our data
df = pd.read_csv("HousePricePrediction.csv", sep=';')
print(df.columns.tolist())

# Remove any rows where the target is missing
df = df[df['SalePrice'].notnull()]

#Remove any white spaces
df.columns = df.columns.str.strip()

#Define x and y
X = df[['MSSubClass', 'MSZoning', 'LotArea', 'LotConfig', 'BldgType', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'Exterior1st', 'BsmtFinSF2', 'TotalBsmtSF']]
y = df['SalePrice']

#Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y_train.isnull().sum())

#Fill in missing target values
y_train = y_train.fillna(y_train.median())

#Fill missing feature values
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)


#Numerical columns
numerical_features = ['MSSubClass', 'LotArea', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF2', 'TotalBsmtSF']
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

#Categorical columns
categorical_features = ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']
X_train[categorical_features] = X_train[categorical_features].astype(str)
X_test[categorical_features] = X_test[categorical_features].astype(str)

#Combine transformations
preprocessor = ColumnTransformer(
    transformers= [
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42))
])

#Beginning training the pipeline
pipeline.fit(X_train, y_train)

#Predict
y_pred = pipeline.predict(X_test)

#Evaluate 
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root MSE: {np.sqrt(mse):.2f}")
print(f"R^2 Score: {r2:.2f}")

#Plot Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Predicted Sale Prices vs Actual Sale Prices')
plt.show()