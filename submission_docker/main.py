import os
import sys
import pandas as pd
import numpy as np
import torch
import logging

try:
    from lib.data_loader import DataLoader
    from lib.feature_engineering import Normalizer, load_normalizer, get_features_final
    from lib.ts_transformer import TSTransformerEncoderClassiregressor
except ImportError as e:
    print(f"Error: Cannot import modules from lib/. {e}")
    print("Please ensure lib/ folder contains __init__.py, data_loader.py, feature_engineering.py, and ts_transformer.py")
    sys.exit(1)

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("PHM-AP 2025 Competition Inference Script Starting")

MODEL_FILENAME = 'model_best.pth'
NORMALIZER_FILENAME = 'normalization.pickle'

MODEL_CONFIG = {
    'feat_dim': 20,
    'max_len': 1500,
    'd_model': 256,
    'n_heads': 16,
    'num_layers': 3,
    'dim_feedforward': 512,
    'num_classes': 1,
    'dropout': 0.1,
    'pos_encoding': 'learnable',
    'activation': 'gelu',
    'norm': 'BatchNorm',
    'freeze': False
}

FEATURE_CONFIG = {
    'num_windows': 1500,
    'columns_to_process': ['Acceleration X (g)', 'Acceleration Y (g)', 'Acceleration Z (g)', 'AE (V)']
}

def load_competition_model(model_path, config):
    logger.info(f"Initializing model with config: {config}")
    try:
        model = TSTransformerEncoderClassiregressor(
            feat_dim=config['feat_dim'],
            max_len=config['max_len'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            num_classes=config['num_classes'],
            dropout=config['dropout'],
            pos_encoding=config['pos_encoding'],
            activation=config['activation'],
            norm=config['norm'],
            freeze=config['freeze']
        )

        logger.info("Loading model weights...")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        logger.info(f"Model weights successfully loaded from {model_path}")
        return model
    except KeyError:
        logger.error(f"Failed to load model: 'state_dict' key not found in .pth file.")
        logger.error("Please check if the .pth file is a standard checkpoint dictionary.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load model. {e}")
        logger.error("Please ensure MODEL_CONFIG exactly matches your saved .pth file.")
        sys.exit(1)

def main():
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    if os.path.exists('/tcdata'):
        logger.info("Docker environment detected (found /tcdata).")
        controller_path = '/tcdata/Controller_Data'
        sensor_path = '/tcdata/Sensor_Data'
        output_path = '/work/result.csv'
        base_model_path = 'model'
    else:
        logger.warning("Docker environment not detected. Switching to local test mode.")
        logger.warning("Ensure you have created 'tcdata' and 'work' folders in the parent directory of 'project/'.")
        controller_path = os.path.normpath('../tcdata/Controller_Data')
        sensor_path = os.path.normpath('../tcdata/Sensor_Data')
        output_path = os.path.normpath('../work/result.csv')
        base_model_path = 'model'

    if not os.path.exists('/tcdata'):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not os.path.exists(controller_path):
            logger.error(f"Local test error: Input data not found at {controller_path}.")
            logger.error("Please create tcdata folder in project's parent directory and add test data.")
            sys.exit(1)

    model_path = os.path.join(base_model_path, MODEL_FILENAME)
    normalizer_path_abs = os.path.join(base_model_path, NORMALIZER_FILENAME)

    logger.info(f"Controller data path: {controller_path}")
    logger.info(f"Sensor data path: {sensor_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Normalizer path: {normalizer_path_abs}")

    try:
        normalizer = load_normalizer(normalizer_path_abs)
    except FileNotFoundError:
        logger.error(f"Fatal error: Normalization file not found: {normalizer_path_abs}")
        logger.error(f"Please ensure training-generated '{NORMALIZER_FILENAME}' is in '{base_model_path}/' folder.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading normalizer: {e}")
        sys.exit(1)

    model = load_competition_model(model_path, MODEL_CONFIG)
    model.to(device)

    loader = DataLoader(
        controller_data_path=controller_path,
        sensor_data_path=sensor_path,
        set_available=[1, 2, 3]
    )

    evalset_list = [1, 2, 3]
    cut_list = list(range(2, 27))

    results = []
    logger.info(f"Starting prediction for {len(evalset_list)} sets and {len(cut_list)} cuts/set...")

    for set_no in evalset_list:
        for cut_no in cut_list:
            logger.info(f'Processing: evalset_{set_no:02d}, Cut {cut_no:02d}...')

            try:
                controller_df = loader.get_controller_data(set_no, cut_no)
                sensor_df = loader.get_sensor_data(set_no, cut_no)

                if sensor_df.empty:
                    logger.warning(f"  -> Sensor data empty (or controller data missing), predicting 0.0.")
                    results.append([f'evalset_{set_no:02d}', cut_no, 0.0])
                    continue

                model_input = get_features_final(
                    controller_df=controller_df,
                    sensor_df=sensor_df,
                    normalizer=normalizer,
                    num_windows=FEATURE_CONFIG['num_windows'],
                    columns_to_process=FEATURE_CONFIG['columns_to_process']
                )

                model_input = model_input.to(device)

                batch_size, seq_len, _ = model_input.shape
                padding_masks = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)

                with torch.no_grad():
                    pred_tensor = model(model_input, padding_masks)

                prediction = pred_tensor.item()
                results.append([f'evalset_{set_no:02d}', cut_no, prediction])
                logger.info(f'  -> Prediction: {prediction:.4f}')

            except Exception as e:
                logger.error(f"  -> Failed processing evalset_{set_no:02d}, Cut {cut_no:02d}: {e}", exc_info=True)
                results.append([f'evalset_{set_no:02d}', cut_no, 0.0])

    logger.info("Prediction loop completed. Saving results...")
    result_df = pd.DataFrame(results, columns=['set_num', 'cut_num', 'pred'])

    try:
        result_df.to_csv(output_path, index=False)
        logger.info(f"Results successfully saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save result.csv to {output_path}: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()