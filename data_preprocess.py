import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# List of sensor columns to be kept for training and testing.
# Based on common practice, columns 1, 5, 6, 10, 16, 18, 19 are often dropped
# due to being constant or having high noise/low correlation.
SENSOR_COLS = [
    's2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14',
    's15', 's17', 's20', 's21', 's22', 's23', 's24'
]
# Column names for the raw data files
COLUMNS = (
    ['unit_nr', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] +
    [f's{i}' for i in range(1, 27)]
)

def load_data(file_name):
    """Loads a C-MAPSS dataset file."""
    # Read the file, replacing internal spaces with a single space, then split by space.
    df = pd.read_csv(
        file_name, sep='\s+', header=None, names=COLUMNS, skipinitialspace=True
    )
    # Drop constant columns (op_setting_3, s1, s5, s6, s10, s16, s18, s19)
    df.drop(columns=['op_setting_3', 's1', 's5', 's6', 's10', 's16', 's18', 's19'], inplace=True)
    return df

def calculate_rul(df):
    """Calculates Remaining Useful Life (RUL) for training data based on max cycles."""
    # Get the max cycle for each engine unit
    max_cycles = df.groupby('unit_nr')['time_cycles'].max()
    
    # Calculate RUL: RUL = Max Cycle for unit - Current Cycle
    def add_rul(row):
        return max_cycles[row['unit_nr']] - row['time_cycles']
    
    df['RUL'] = df.apply(add_rul, axis=1)
    
    # The C-MAPSS problem often caps RUL at 125 cycles
    df['RUL'] = df['RUL'].clip(upper=125)
    return df

def normalize_data(train_df, test_df):
    """Normalizes the sensor data using MinMaxScaler fit only on the training set."""
    scaler = MinMaxScaler()
    
    # Fit scaler only on the training set sensor columns
    scaler.fit(train_df[SENSOR_COLS])
    
    # Apply transformation to both training and test sets
    train_df[SENSOR_COLS] = scaler.transform(train_df[SENSOR_COLS])
    test_df[SENSOR_COLS] = scaler.transform(test_df[SENSOR_COLS])
    
    return train_df, test_df, scaler

def create_sequences(X, sequence_length):
    """Converts a feature array into overlapping sequences for the VAE."""
    sequences = []
    for i in range(len(X) - sequence_length + 1):
        sequences.append(X[i:i + sequence_length])
    return np.array(sequences)

def prepare_train_sequences(train_df, sequence_length):
    """
    Prepares training sequences: only uses data from the 'healthy'
    initial phase (RUL > 125) to train the VAE on normal condition.
    """
    # Filter for 'healthy' operating cycles (RUL > 125 cycles)
    # This assumes that the failure trend starts after 125 cycles
    healthy_data = train_df[train_df['RUL'] == 125]
    
    # Create sequences for all units
    all_sequences = []
    for unit_nr in healthy_data['unit_nr'].unique():
        unit_data = healthy_data[healthy_data['unit_nr'] == unit_nr]
        features = unit_data[SENSOR_COLS].values
        
        # Only create sequences if the unit data is long enough
        if len(features) >= sequence_length:
            sequences = create_sequences(features, sequence_length)
            all_sequences.append(sequences)
            
    # Concatenate all sequences
    X_train_sequences = np.concatenate(all_sequences, axis=0)
    
    # VAE input and output are the same (reconstruction)
    return X_train_sequences, X_train_sequences


def prepare_test_sequences(test_df, sequence_length):
    """Prepares test sequences for anomaly detection."""
    X_test_sequences = []
    
    # Iterate over each engine unit
    for unit_nr in test_df['unit_nr'].unique():
        unit_data = test_df[test_df['unit_nr'] == unit_nr]
        features = unit_data[SENSOR_COLS].values
        
        # Create sequences for the entire time series of the unit
        if len(features) >= sequence_length:
            sequences = create_sequences(features, sequence_length)
            X_test_sequences.append(sequences)
        else:
            # Handle short sequences by padding or skipping (skipping here)
            print(f"Warning: Unit {unit_nr} in test set is too short ({len(features)} cycles) to form sequences of length {sequence_length}. Skipping.")

    # Concatenate all sequences
    X_test_sequences = np.concatenate(X_test_sequences, axis=0)
    
    return X_test_sequences