import pandas as pd
import numpy as np
import torch
import pickle
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)

# --- ANOMALY REMOVAL FUNCTION ---

def remove_anomalies_iqr(df: pd.DataFrame, columns_to_process: list, k: float = 6.0) -> pd.DataFrame:
    """
    Removes extreme outliers from sensor data using the Interquartile Range (IQR).
    Uses clipping to maintain the original shape of the DataFrame.
    """
    # This .copy() is crucial to avoid SettingWithCopyWarning
    df_cleaned = df.copy() 
    
    for col in columns_to_process:
        if col in df_cleaned.columns:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - (k * IQR)
            upper_bound = Q3 + (k * IQR)
            
            df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
            
    return df_cleaned

# --- Normalizer Class ---

class Normalizer(object):
    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None):
        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        if self.norm_type == "standardization":
            if self.mean is None:
                raise ValueError("Mean is None, normalizer not loaded correctly")
            return (df - self.mean) / (self.std + np.finfo(float).eps)
        elif self.norm_type == "minmax":
            if self.max_val is None:
                raise ValueError("Max/Min is None, normalizer not loaded correctly")
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)
        else:
            raise NameError(f'Normalize method "{self.norm_type}" not implemented')

# --- Statistical Functions ---

def kurt(x: pd.Series) -> float:
    if x.empty:
        return np.nan
    return x.kurtosis()

def rms(x: pd.Series) -> float:
    if x.empty:
        return np.nan
    return np.sqrt(np.mean(x ** 2))

def crest_factor(x: pd.Series) -> float:
    if x.empty:
        return np.nan
    x_rms = rms(x)
    if x_rms == 0:
        return np.nan
    x_peak_abs = x.abs().max()
    return x_peak_abs / x_rms

# --- Feature Engineering Function ---

def create_window_features(df_cut: pd.DataFrame, num_windows: int, columns_to_process: list) -> pd.DataFrame:
    stats_to_calc = [rms, 'max', 'std', kurt, crest_factor]
    feat_names = [stat.__name__ if hasattr(stat, '__name__') else stat for stat in stats_to_calc]

    valid_cols = [col for col in columns_to_process if col in df_cut.columns]

    if df_cut.empty or not valid_cols:
        if df_cut.empty:
            logger.warning("Sensor dataframe passed to create_window_features is empty.")
        else:
            logger.error(f"Specified columns {columns_to_process} not found in DataFrame.")
        all_feat_names = [f"{col}_{feat}" for col in columns_to_process for feat in feat_names]
        return pd.DataFrame(np.zeros((num_windows, len(all_feat_names))), columns=all_feat_names)

    L = len(df_cut)
    if L < num_windows:
        logger.warning(f"DataFrame length ({L}) is less than 'num_windows' ({num_windows}). Using {L} windows.")
        num_windows_actual = L
    else:
        num_windows_actual = num_windows

    group_id = np.floor(np.arange(L) * num_windows_actual / L).astype(int)
    features_df = df_cut.groupby(group_id)[valid_cols].agg(stats_to_calc)

    if isinstance(features_df.columns, pd.MultiIndex):
        new_column_names = [f"{col[0]}_{col[1]}" for col in features_df.columns]
        features_df.columns = new_column_names

    all_feat_names = [f"{col}_{feat}" for col in columns_to_process for feat in feat_names]
    for col_name in all_feat_names:
        if col_name not in features_df:
            logger.warning(f"Feature column {col_name} not generated (sensor data missing), filling with 0.")
            features_df[col_name] = 0.0

    features_df = features_df[all_feat_names]

    if L < num_windows:
        logger.info(f"Padding features from {L} windows to {num_windows} windows.")
        pad_width = num_windows - L
        padding_df = pd.DataFrame(0, index=range(pad_width), columns=features_df.columns)
        features_df = pd.concat([features_df, padding_df], ignore_index=True)

    if len(features_df) > num_windows:
        features_df = features_df.iloc[:num_windows]

    features_df.index.name = 'window_id'
    return features_df

# --- Helper Function ---

def load_normalizer(filepath: str) -> Normalizer:
    logger.info(f"Loading normalizer from {filepath}...")
    try:
        with open(filepath, 'rb') as f:
            norm_dict = pickle.load(f)

        normalizer = Normalizer(**norm_dict)
        logger.info(f"Normalizer loaded successfully (type: {normalizer.norm_type})")
        if hasattr(normalizer, 'mean'):
            logger.info(f"  - Found {len(normalizer.mean)} means. Columns: {normalizer.mean.index[:3]}...")
        return normalizer
    except FileNotFoundError:
        logger.error(f"Fatal error: Normalization file not found: {filepath}")
        logger.error("Please ensure training-generated 'normalization.pickle' is in 'model/' folder.")
        raise
    except Exception as e:
        logger.error(f"Error loading 'normalization.pickle': {e}")
        raise

# --- MAIN PREDICTION FUNCTION ---

def get_features_final(controller_df: pd.DataFrame,
                       sensor_df: pd.DataFrame,
                       normalizer: Normalizer,
                       num_windows: int,
                       columns_to_process: list) -> torch.FloatTensor:
    
    # 1. Apply Anomaly Removal (k=6.0)
    logger.debug(f"Applying anomaly removal (k=6.0) to {len(sensor_df)} sensor readings.")
    sensor_df_cleaned = remove_anomalies_iqr(
        df=sensor_df,
        columns_to_process=columns_to_process, 
        k=6.0
    )

    # 2. Create Window Features (using cleaned data)
    feature_df = create_window_features(
        df_cut=sensor_df_cleaned, # Use the cleaned DataFrame
        # df_cut=sensor_df,
        num_windows=num_windows,
        columns_to_process=columns_to_process
    )

    # 3. Rename columns for consistency
    if feature_df.shape[1] == 20:
        # Use generic names that match the normalizer's expected names
        generic_names = [f'dim_{i}' for i in range(feature_df.shape[1])]
        feature_df.columns = generic_names
    else:
        logger.error(f"Feature engineering generated {feature_df.shape[1]} features instead of 20.")
        raise ValueError("Feature count mismatch")

    # 4. Re-order columns to match normalizer
    try:
        if hasattr(normalizer, 'mean') and normalizer.mean is not None:
            ordered_columns = normalizer.mean.index
            feature_df = feature_df[ordered_columns]
        elif hasattr(normalizer, 'min_val') and normalizer.min_val is not None:
            ordered_columns = normalizer.min_val.index
            feature_df = feature_df[ordered_columns]
        else:
            raise ValueError("No 'mean' or 'min_val' index found in normalizer")
    except KeyError:
        logger.error("Feature engineering columns do not match normalizer columns.")
        logger.error(f"Normalizer expects: {list(ordered_columns)}")
        logger.error(f"Actual generated: {list(feature_df.columns)}")
        raise

    # 5. Normalize and Convert to Tensor
    normalized_df = normalizer.normalize(feature_df)
    features_tensor = torch.FloatTensor(normalized_df.values).unsqueeze(0)

    return features_tensor