import pandas as pd
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def evaluate(model_dir: str, data_path: str):
    if not os.path.exists(data_path) or not os.path.exists(model_dir):
        print("Paths not found.")
        return

    df = pd.read_csv(data_path)
    X = df.drop(columns=['power', 'delay', 'area'])

    for target in ['power', 'delay', 'area']:
        model_path = os.path.join(model_dir, f'rf_{target}.joblib')
        if not os.path.exists(model_path):
            continue

        model = joblib.load(model_path)
        y = df[target]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)

        print(f"Target: {target.capitalize()} | R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

if __name__ == "__main__":
    evaluate('models/weights', 'decoder_power_delay_area_dataset.csv')
