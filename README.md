🏡 House Price Prediction using XGBoost & Scikit-Learn Pipelines
This project demonstrates how to predict house prices using XGBoost Regressor in combination with Scikit-Learn Pipelines for preprocessing both numerical and categorical features. The workflow covers data cleaning, preprocessing, model training, evaluation, and visualization.

📂Dataset
The dataset HousePricePrediction.csv contains various house features such as:

MSSubClass

MSZoning

LotArea

LotConfig

BldgType

OverallCond

YearBuilt

YearRemodAdd

Exterior1st

BsmtFinSF2

TotalBsmtSF

SalePrice (target variable)

🚀 Workflow Summary
1️⃣ Data Loading and Cleaning
Loaded data with pandas.

Removed unnecessary whitespaces from column names.

Handled missing target values by filtering out rows with missing SalePrice.

Filled missing feature values with 0.

2️⃣ Feature Engineering
Split features into:

Numerical columns: scaled using StandardScaler.

Categorical columns: encoded using OneHotEncoder.

Used ColumnTransformer to combine preprocessing steps.

3️⃣ Model Building
Built a Pipeline:

Preprocessing (StandardScaler, OneHotEncoder)

Model (XGBRegressor with reg:squarederror objective)

4️⃣ Model Training
Trained the pipeline on 80% of the data (using train_test_split).

Predictions were made on the 20% test set.

5️⃣ Evaluation Metrics
Evaluated model performance using:

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R² Score

6️⃣ Visualization
Visualized Actual vs Predicted sale prices using matplotlib to assess the model's prediction quality.

📊 Results
Metric	Value
MSE	~1,584,411,917
RMSE	~39,804
R²	~0.79

A scatter plot showed a good alignment between predicted and actual sale prices, confirming reasonable model accuracy.

📈 Key Takeaways
XGBoost works well with a pipeline integrating both categorical and numerical preprocessing.

Proper data cleaning (e.g., removing NaNs) is critical.

Pipeline architecture simplifies experimentation and future adjustments.
