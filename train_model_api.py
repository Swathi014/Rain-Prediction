import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("Starting API-compatible model training script...")

# --- 1. Define Features ---
# These features match the OpenWeatherMap API
api_features = [
    'MinTemp',     # OWM 'temp_min'
    'MaxTemp',     # OWM 'temp_max'
    'Rainfall',    # OWM 'rain.1h'
    'WindSpeed3pm',# OWM 'wind.speed'
    'Humidity3pm', # OWM 'main.humidity'
    'Pressure3pm', # OWM 'main.pressure'
    'Cloud3pm',    # OWM 'clouds.all'
    'RainToday',   # Manual input
    'RainTomorrow' # Target
]

# --- 2. Load Data ---
try:
    df = pd.read_csv('weatherAUS.csv', usecols=api_features)
    print(f"Dataset 'weatherAUS.csv' loaded.")
except FileNotFoundError:
    print("Error: 'weatherAUS.csv' not found.")
    print("Please download it from Kaggle and place it in the same directory.")
    exit()

# --- 3. Preprocessing ---
df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})
df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1})
df = df.dropna(subset=['RainTomorrow', 'RainToday'])

X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']

# --- 4. Impute Missing Data ---
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
print("Data imputation complete.")

# --- 5. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 6. Feature Scaling ---
# We fit the scaler ONLY on the training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Data scaling complete.")

# --- 7. Train Model ---
print("Training Random Forest model...")
api_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
api_model.fit(X_train, y_train)
print("Model training complete.")

# --- 8. Save the 3 Files ---
joblib.dump(api_model, 'api_model.pkl')
joblib.dump(scaler, 'api_scaler.pkl')
joblib.dump(X.columns, 'api_model_columns.pkl')

print("-------------------------------------------------")
print("SUCCESS: Your 3 files have been saved:")
print("1. api_model.pkl")
print("2. api_scaler.pkl")
print("3. api_model_columns.pkl")
print("-------------------------------------------------")