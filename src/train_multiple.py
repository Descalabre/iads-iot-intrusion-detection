"""
Script to train the IADS model using all CSV files in a specified folder.
"""

import os
import sys
import pandas as pd
from glob import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import IADSController

def train_with_folder(folder_path, model_name):
    # Find all CSV files in the folder
    csv_files = glob(os.path.join(folder_path, '*.csv'))
    if not csv_files:
        raise ValueError(f"No CSV files found in folder: {folder_path}")
    
    data_frames = []
    for file_path in csv_files:
        df = pd.read_csv(file_path, low_memory=False)
        # Convert all columns except target to numeric, coercing errors to NaN
        for col in df.columns:
            if col != 'label':  # Assuming 'label' is the target column
                df[col] = pd.to_numeric(df[col], errors='coerce')
        data_frames.append(df)
    
    combined_data = pd.concat(data_frames, ignore_index=True)
    
    # Save combined data temporarily
    temp_combined_path = os.path.join(folder_path, "combined_training_data.csv")
    combined_data.to_csv(temp_combined_path, index=False)
    
    # Initialize controller and train
    controller = IADSController()
    controller.train(temp_combined_path, model_name=model_name)
    
    # Optionally remove temp file after training
    # os.remove(temp_combined_path)

if __name__ == "__main__":
    folder_path = "data/processed"
    model_name = "betav1"
    train_with_folder(folder_path, model_name)
