import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. Load and preprocess data
df = pd.read_excel('data_collection.xlsx', sheet_name='Sheet1', nrow=2000)
df.columns = df.columns.str.strip()
df['BPA_Level'] = pd.to_numeric(df['GM'], errors='coerce')
df = df.dropna(subset=['BPA_Level'])

num_cols = [...]  # numeric feature names
cat_cols = [...]  # categorical feature names

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
for col in cat_cols:
    df[col] = df[col].astype(str).fillna('Missing')

X = df[num_cols + cat_cols]
y = df['BPA_Level']

# 2. Pipeline: preprocessing + model
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# 3. Train/test split and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluation
y_pred = model.predict(X_test)
print(f"R²: {r2_score(y_test, y_pred):.3f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
