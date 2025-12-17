import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer

# Paths relative to the root 'neurobed_project' folder
MODEL_PATH = "backend/los_model.pkl"
DATA_PATH = "data/synthetic_rule_based.csv"

class MLEngine:
    def __init__(self):
        self.pipeline = None
        self.load_or_train_model()

    def load_or_train_model(self):
        """
        Tries to load a saved model. If not found, trains a new one.
        """
        if os.path.exists(MODEL_PATH):
            try:
                self.pipeline = joblib.load(MODEL_PATH)
                print("✅ Model loaded from disk.")
            except:
                print("⚠️ Model file corrupted. Retraining...")
                self.train_model()
        else:
            print("⚠️ Model not found. Training new model...")
            self.train_model()

    def train_model(self):
        """
        Loads data (or creates dummy data), trains the model, and saves it.
        """
        # 1. Load Data
        if os.path.exists(DATA_PATH):
            print(f"📂 Loading data from {DATA_PATH}...")
            df = pd.read_csv(DATA_PATH)
            # Identify the target column dynamically
            if 'los_days_ceiled' in df.columns:
                target = 'los_days_ceiled'
            elif 'los_days' in df.columns:
                target = 'los_days'
            else:
                target = 'los' # Fallback
                df['los'] = np.random.uniform(1, 15, len(df))
        else:
            print("⚠️ CSV not found. Generatng synthetic training data...")
            # Fallback: Generate dummy data if CSV is missing
            df = pd.DataFrame({
                'age': np.random.randint(18, 95, 500),
                'base_severity': np.random.randint(1, 6, 500),
                'sofa': np.random.uniform(0, 18, 500),
                'cci': np.random.randint(0, 12, 500),
                'infection_flag': np.random.randint(0, 2, 500),
                'admission_type': np.random.choice(['Emergency', 'Elective'], 500),
                'los_days': np.random.uniform(1, 20, 500)
            })
            target = 'los_days'

        # 2. Prepare Features (X) and Target (y)
        # We ensure these columns exist to avoid KeyErrors
        required_cols = ['age', 'base_severity', 'sofa', 'cci', 'infection_flag', 'admission_type']
        
        # Fill missing columns with defaults if they don't exist in CSV
        for col in required_cols:
            if col not in df.columns:
                if col == 'admission_type':
                    df[col] = 'Emergency'
                else:
                    df[col] = 0

        X = df[required_cols].copy()
        y = df[target]

        # 3. Define Preprocessing Pipeline
        numeric_features = ['age', 'base_severity', 'sofa', 'cci', 'infection_flag']
        categorical_features = ['admission_type']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # 4. Train Model
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
        
        self.pipeline.fit(X, y)
        
        # 5. Save Model
        joblib.dump(self.pipeline, MODEL_PATH)
        print("✅ Model trained and saved to backend/los_model.pkl")

    def predict(self, data: dict):
        """
        Accepts a dictionary of patient data, returns LOS (float).
        """
        # Convert single dict to DataFrame
        df_in = pd.DataFrame([data])
        
        # Ensure all required columns exist (fill with 0 if missing)
        required_cols = ['age', 'base_severity', 'sofa', 'cci', 'infection_flag', 'admission_type']
        for col in required_cols:
            if col not in df_in.columns:
                if col == 'admission_type':
                    df_in[col] = 'Emergency'
                else:
                    df_in[col] = 0
        
        # Predict
        prediction = self.pipeline.predict(df_in)[0]
        
        # Return a reasonable positive number (at least 1 day)
        return max(1.0, round(prediction, 1))
    #new comment