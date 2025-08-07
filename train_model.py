import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
import joblib
import os
from utils.question_bank import QUESTIONS

# Create directories if they don't exist
os.makedirs('model', exist_ok=True)

# Load the dataset
try:
    print("Loading dataset...")
    # The file is tab-separated, and the first row is the header.
    data = pd.read_csv('data/ipip-50-dataset.csv', sep='\t', header=0, on_bad_lines='warn', engine='python')
    print(f"Dataset loaded. Shape: {data.shape}")

    # Clean column names by stripping whitespace
    data.columns = data.columns.str.strip()
    print("Column names cleaned.")

    # Get the list of all 50 question columns from the question bank
    q_cols = list(QUESTIONS.keys())
    print(f"Expecting columns: {q_cols}")
    
    # Ensure all expected question columns are present
    missing_cols = [col for col in q_cols if col not in data.columns]
    if missing_cols:
        print(f"Error: The following columns are missing from the dataset: {missing_cols}")
        exit()

    # Filter the dataframe to only include the 50 question columns
    data = data[q_cols]
    print("Filtered to 50 question columns.")
    
    # Convert all question columns to numeric, coercing errors to NaN
    for col in q_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    print("Converted all question columns to numeric.")
    
    # Drop rows with any NaN values
    data.dropna(inplace=True)
    print(f"Rows after dropping NaNs: {data.shape[0]}")
    
    # Convert valid question columns to integers
    data[q_cols] = data[q_cols].astype(int)
    print("Converted question columns to int.")

    # Check if the dataframe is empty after cleaning
    if data.empty:
        print("Error: Dataframe is empty after cleaning. The data could not be parsed correctly.")
        exit()

except FileNotFoundError:
    print("Error: Dataset not found. Make sure 'ipip-50-dataset.csv' is in the 'data' directory.")
    exit()
except Exception as e:
    print(f"An error occurred while reading the data: {e}")
    exit()

# Define OCEAN traits and their question mappings (including reversed questions)
OCEAN_MAP = {
    'EXT': ['EXT1', 'EXT3', 'EXT5', 'EXT7', 'EXT9'],
    'EXT_R': ['EXT2', 'EXT4', 'EXT6', 'EXT8', 'EXT10'],
    'EST': ['EST1', 'EST3', 'EST5', 'EST6', 'EST7', 'EST8', 'EST9', 'EST10'],
    'EST_R': ['EST2', 'EST4'],
    'AGR': ['AGR2', 'AGR4', 'AGR6', 'AGR8', 'AGR9', 'AGR10'],
    'AGR_R': ['AGR1', 'AGR3', 'AGR5', 'AGR7'],
    'CSN': ['CSN1', 'CSN3', 'CSN5', 'CSN7', 'CSN9', 'CSN10'],
    'CSN_R': ['CSN2', 'CSN4', 'CSN6', 'CSN8'],
    'OPN': ['OPN1', 'OPN3', 'OPN5', 'OPN7', 'OPN8', 'OPN9', 'OPN10'],
    'OPN_R': ['OPN2', 'OPN4', 'OPN6']
}

# Calculate OCEAN scores
for trait, questions in OCEAN_MAP.items():
    if '_R' in trait:
        # Reverse score for reverse-keyed items (6 - value)
        for question in questions:
            data[question] = 6 - data[question]

# Define final trait columns
data['EXT'] = data[OCEAN_MAP['EXT'] + OCEAN_MAP['EXT_R']].mean(axis=1)
data['EST'] = data[OCEAN_MAP['EST'] + OCEAN_MAP['EST_R']].mean(axis=1)
data['AGR'] = data[OCEAN_MAP['AGR'] + OCEAN_MAP['AGR_R']].mean(axis=1)
data['CSN'] = data[OCEAN_MAP['CSN'] + OCEAN_MAP['CSN_R']].mean(axis=1)
data['OPN'] = data[OCEAN_MAP['OPN'] + OCEAN_MAP['OPN_R']].mean(axis=1)

# Feature Selection
X = data.iloc[:, 0:50]
y = data[['EXT', 'EST', 'AGR', 'CSN', 'OPN']]

# Use mutual information to find the best 10 features
importances = mutual_info_regression(X, y.sum(axis=1))
feature_importances = pd.Series(importances, index=X.columns)
selected_features = feature_importances.nlargest(10).index.tolist()

print("Selected Features:", selected_features)

# Save the selected features
joblib.dump(selected_features, 'model/feature_selector.pkl')

# --- Train OCEAN Prediction Model ---
X_selected = data[selected_features]
y_ocean = data[['EXT', 'EST', 'AGR', 'CSN', 'OPN']]

X_train, X_test, y_train, y_test = train_test_split(X_selected, y_ocean, test_size=0.2, random_state=42)

ocean_model = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=5)
ocean_model.fit(X_train, y_train)

print("OCEAN Model Score:", ocean_model.score(X_test, y_test))
joblib.dump(ocean_model, 'model/ocean_model.pkl')

# --- Train Imputation Model to predict the other 40 questions ---
all_features = X.columns.tolist()
imputation_target_features = [feat for feat in all_features if feat not in selected_features]

X_imputation_train = data[selected_features]
y_imputation_target = data[imputation_target_features]

imputation_model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=10)
imputation_model.fit(X_imputation_train, y_imputation_target)

print("Imputation model trained.")
joblib.dump(imputation_model, 'model/imputation_model.pkl')

print("Models and feature selector saved successfully in the 'model/' directory.")
