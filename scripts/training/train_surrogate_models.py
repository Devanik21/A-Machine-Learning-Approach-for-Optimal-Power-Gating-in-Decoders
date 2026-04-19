import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_rf_models(data_path: str, output_dir: str):
    """Trains Random Forest surrogate models for Power, Delay, and Area."""
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    X = df.drop(columns=['power', 'delay', 'area'])
    y_power = df['power']
    y_delay = df['delay']
    y_area = df['area']

    X_train, X_test, yp_train, yp_test = train_test_split(X, y_power, test_size=0.2, random_state=42)
    _, _, yd_train, yd_test = train_test_split(X, y_delay, test_size=0.2, random_state=42)
    _, _, ya_train, ya_test = train_test_split(X, y_area, test_size=0.2, random_state=42)

    rf_power = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_delay = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_area = RandomForestRegressor(n_estimators=100, random_state=42)

    rf_power.fit(X_train, yp_train)
    rf_delay.fit(X_train, yd_train)
    rf_area.fit(X_train, ya_train)

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(rf_power, os.path.join(output_dir, 'rf_power.joblib'))
    joblib.dump(rf_delay, os.path.join(output_dir, 'rf_delay.joblib'))
    joblib.dump(rf_area, os.path.join(output_dir, 'rf_area.joblib'))

    print("Models trained and serialized.")

if __name__ == "__main__":
    train_rf_models('decoder_power_delay_area_dataset.csv', 'models/weights')
