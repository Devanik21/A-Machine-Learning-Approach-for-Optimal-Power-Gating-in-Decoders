import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_data(input_path: str, output_path: str):
    """Cleans the raw SPICE simulation dataset."""
    logging.info(f"Loading data from {input_path}")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logging.error("File not found. Please verify the input path.")
        return

    # Drop any duplicates that might have occurred during batch simulations
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    logging.info(f"Dropped {initial_shape[0] - df.shape[0]} duplicate rows.")

    # Handle missing values (if any)
    if df.isnull().sum().sum() > 0:
        logging.warning("Missing values detected. Dropping incomplete records.")
        df.dropna(inplace=True)

    # Physical constraint validation (e.g., power and delay must be positive)
    invalid_rows = df[(df['power'] <= 0) | (df['delay'] <= 0) | (df['area'] <= 0)]
    if not invalid_rows.empty:
        logging.warning(f"Found {len(invalid_rows)} rows with non-positive targets. Removing.")
        df = df[(df['power'] > 0) & (df['delay'] > 0) & (df['area'] > 0)]

    df.to_csv(output_path, index=False)
    logging.info(f"Cleaned dataset saved to {output_path} with shape {df.shape}")

if __name__ == "__main__":
    clean_data('decoder_power_delay_area_dataset.csv', 'data_processing/clean_dataset.csv')
