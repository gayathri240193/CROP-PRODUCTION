# train_model.py

import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Load cleaned and pivoted dataset
df = pd.read_csv("Crop_Production_Cleaned.csv")

pivot_df = df.pivot_table(index=['Area', 'Item', 'Year'], columns='Element', values='Value').reset_index()
pivot_df.dropna(subset=['Production', 'Yield', 'Area harvested'], inplace=True)

X = pivot_df[['Area', 'Item', 'Year', 'Yield', 'Area harvested']]
y = pivot_df['Production']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical = ['Area', 'Item']
numerical = ['Year', 'Yield', 'Area harvested']

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical)
], remainder='passthrough')

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
])

pipeline.fit(X_train, y_train)

with open("final_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved as final_model.pkl")
