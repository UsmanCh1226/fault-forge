import pandas as pd
import numpy as np

# Column names based on the C-MAPSS readme.txt
COLUMNS = [
    'unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6',
    'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12',
    'sensor_13', 'sensor_14', 'sensor_15', 'sensor_16', 'sensor_17', 'sensor_18',
    'sensor_19', 'sensor_20', 'sensor_21'
]

# Sensors and settings that are constant or have minimal variance, and thus often dropped
DROP_COLS = ['op_setting_3', 'sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']

# Sensor and Operational Setting columns that are kept for features
FEATURE_COLS = [
    'op_setting_1', 'op_setting_2', 'sensor_2', 'sensor_3', 'sensor_4', 
    'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 
    'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'
]

def load_and_clean_data(filepath):
    """
    Loads the C-MAPSS data file, assigns column names, and drops specified columns.
    
    Args:
        filepath (str): Path to the training or test data file.

    Returns:
        pd.DataFrame: The loaded and cleaned DataFrame.
    """
    # Load data using space as separator and handling missing values
    df = pd.read_csv(filepath, sep="\\s+", header=None, names=COLUMNS, skipinitialspace=True)
    
    # Drop the constant/redundant columns
    df = df.drop(columns=DROP_COLS)
    
    return df

def calculate_train_rul(df):
    """
    Calculates Remaining Useful Life (RUL) for the training data set.
    RUL = (Max Cycles for engine) - (Current Cycle).
    """
    # Find the maximum 'time_in_cycles' for each engine unit
    max_cycles = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycle']
    
    # Merge the max cycle information back into the main dataframe
    df = df.merge(max_cycles, on='unit_number', how='left')
    
    # Calculate RUL
    df['RUL'] = df['max_cycle'] - df['time_in_cycles']
    
    # The 'max_cycle' column is no longer needed
    df = df.drop(columns=['max_cycle'])
    
    return df

def load_test_rul(r_filepath):
    """
    Loads the true RUL values for the end of the test trajectories. 
    These values represent the RUL at the LAST recorded cycle for each engine.
    """
    # The RUL file contains one RUL value per engine trajectory's end cycle
    # It has a second empty column, so we read only the first
    true_rul_df = pd.read_csv(r_filepath, sep='\\s+', header=None, names=['RUL_true'])
    return true_rul_df['RUL_true'].values

def apply_piecewise_rul(df, cap=125):
    """
    Applies a piecewise linear RUL model, capping the RUL value at a maximum.
    This is standard for C-MAPSS datasets, where RUL is considered constant 
    when the engine is very healthy. 
    """
    df['RUL_capped'] = np.minimum(df['RUL'], cap)
    return df

def normalize_data(df_train, df_test):
    """
    Normalizes the feature columns based on the training data's mean and std deviation.
    The same scaler is applied to the test data to prevent data leakage.
    """
    df_train_norm = df_train.copy()
    df_test_norm = df_test.copy()
    
    # Calculate mean and standard deviation ONLY from the training data
    scaler_mean = df_train[FEATURE_COLS].mean()
    scaler_std = df_train[FEATURE_COLS].std()
    
    # Handle zero standard deviation
    scaler_std[scaler_std == 0] = 1.0

    # Apply standardization: (x - mean) / std
    df_train_norm[FEATURE_COLS] = (df_train_norm[FEATURE_COLS] - scaler_mean) / scaler_std
    df_test_norm[FEATURE_COLS] = (df_test_norm[FEATURE_COLS] - scaler_mean) / scaler_std
    
    print("\nFeatures normalized based on training data statistics.")
    
    return df_train_norm, df_test_norm

def create_sequences(df, sequence_length, feature_cols, target_col=None):
    """
    Converts the time-series data into sequences of a fixed length for an RNN/LSTM model.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        sequence_length (int): The number of time steps in each sequence.
        feature_cols (list): List of columns to use as features.
        target_col (str, optional): The column to use as the target (e.g., 'RUL_capped').

    Returns:
        X (np.array): Sequences of features.
        y (np.array, optional): Target values corresponding to the end of each sequence.
        engine_indices (np.array): Original unit numbers for each sequence.
    """
    X, y, engine_indices = [], [], []
    
    for unit_number in df['unit_number'].unique():
        unit_df = df[df['unit_number'] == unit_number]
        
        # Skip units that are too short for the sequence length
        if len(unit_df) < sequence_length:
            continue
            
        unit_data = unit_df[feature_cols].values
        
        for i in range(len(unit_df) - sequence_length + 1):
            # Create a sequence of 'sequence_length' steps
            X.append(unit_data[i:i + sequence_length])
            
            # The target is the RUL at the END of the sequence
            if target_col:
                unit_targets = unit_df[target_col].values
                y.append(unit_targets[i + sequence_length - 1])
            
            # Store the index of the engine corresponding to the sequence's end cycle
            engine_indices.append(unit_number)
                
    X = np.array(X)
    engine_indices = np.array(engine_indices)
    
    if target_col:
        y = np.array(y)
        return X, y, engine_indices
    else:
        return X, engine_indices

def calculate_phm_score(y_true, y_pred):
    """
    Calculates the C-MAPSS competition Score (PHM Scoring Function).
    A lower score is better. Penalizes late predictions much more severely.
    """
    # d_i = Predicted RUL - True RUL
    d_i = y_pred - y_true
    
    # Calculate penalty components
    # Positive error penalty (late prediction): exp(d_i/10) - 1
    positive_score = np.sum(np.exp(d_i[d_i >= 0] / 10.0) - 1)
    
    # Negative error penalty (early prediction): exp(-d_i/13) - 1
    negative_score = np.sum(np.exp(-d_i[d_i < 0] / 13.0) - 1)
    
    total_score = positive_score + negative_score
    return total_score