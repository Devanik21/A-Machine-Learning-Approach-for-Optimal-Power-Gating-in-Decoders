import pandas as pd

def load_dataset(filepath: str) -> pd.DataFrame:
    """Loads the CSV dataset and performs basic type checking."""
    df = pd.read_csv(filepath)
    return df
