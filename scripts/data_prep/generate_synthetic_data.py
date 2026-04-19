import numpy as np
import pandas as pd

def generate_synthetic_vlsi_data(num_samples=1000):
    """Generates synthetic data mirroring the VLSI decoder design space for testing purposes."""
    np.random.seed(42)

    decoder_size = np.random.randint(2, 7, num_samples)
    tech_node = np.random.choice([180, 130, 90, 65, 45, 32, 22], num_samples)
    supply_voltage = np.random.uniform(0.6, 1.8, num_samples)
    threshold_voltage = np.random.uniform(0.2, 0.5, num_samples)
    transistor_width = np.random.uniform(0.5, 10.0, num_samples)
    load_capacitance = np.random.uniform(10.0, 200.0, num_samples)
    pg_efficiency = np.random.uniform(0.5, 0.95, num_samples)
    switching_activity = np.random.uniform(0.1, 0.8, num_samples)
    leakage_factor = np.random.uniform(0.01, 0.1, num_samples)
    temperature = np.random.uniform(25.0, 85.0, num_samples)

    # Synthetic target generation based on physical intuition
    power = (switching_activity * load_capacitance * (supply_voltage**2)) + (leakage_factor * supply_voltage) + np.random.normal(0, 0.1, num_samples)
    delay = (load_capacitance * supply_voltage) / (transistor_width * ((supply_voltage - threshold_voltage)**2)) + np.random.normal(0, 0.05, num_samples)
    area = (transistor_width * decoder_size * 2) + np.random.normal(0, 0.5, num_samples)

    # Ensure positivity
    power = np.abs(power)
    delay = np.abs(delay)
    area = np.abs(area)

    df = pd.DataFrame({
        'decoder_size': decoder_size, 'tech_node': tech_node, 'supply_voltage': supply_voltage,
        'threshold_voltage': threshold_voltage, 'transistor_width': transistor_width,
        'load_capacitance': load_capacitance, 'pg_efficiency': pg_efficiency,
        'switching_activity': switching_activity, 'leakage_factor': leakage_factor,
        'temperature': temperature, 'power': power, 'delay': delay, 'area': area
    })
    return df

if __name__ == "__main__":
    df_synthetic = generate_synthetic_vlsi_data(500)
    df_synthetic.to_csv('data_processing/synthetic_decoder_data.csv', index=False)
    print("Synthetic data generated.")
