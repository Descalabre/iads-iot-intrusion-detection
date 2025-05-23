"""
Data Preprocessing module for the Intrusion and Anomaly Detection System (IADS).
Handles data loading, cleaning, feature extraction, and transformation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os
from datetime import datetime

from src.core.config import logger, DATA_DIR, MODELS_DIR

class DataPreprocessor:
    """Handles all data preprocessing operations for the IADS system."""
    
    def __init__(self):
        """Initialize the DataPreprocessor with default parameters."""
        self.scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None
        self.feature_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.numerical_columns: List[str] = []
        self.target_column: Optional[str] = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from various file formats (CSV, Excel, Parquet).
        
        Args:
            file_path: Path to the data file
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
        """
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.csv':
                data = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                data = pd.read_excel(file_path)
            elif file_extension == '.parquet':
                data = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            logger.info(f"Successfully loaded data from {file_path}")
            logger.debug(f"Data shape: {data.shape}")
            return data
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def analyze_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the dataset and return basic statistics.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dict containing analysis results
        """
        try:
            analysis = {
                'shape': data.shape,
                'missing_values': data.isnull().sum().to_dict(),
                'dtypes': data.dtypes.to_dict(),
                'numerical_stats': data.describe().to_dict(),
                'categorical_columns': data.select_dtypes(include=['object']).columns.tolist(),
                'numerical_columns': data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            }
            
            logger.info("Data analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing data: {str(e)}")
            raise

    def clean_data(self, data: pd.DataFrame, 
                  handle_missing: bool = True,
                  remove_duplicates: bool = True,
                  handle_outliers: bool = True) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values, duplicates, and outliers.
        
        Args:
            data: Input DataFrame
            handle_missing: Whether to handle missing values
            remove_duplicates: Whether to remove duplicate rows
            handle_outliers: Whether to handle outliers
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        try:
            df = data.copy()
            
            if handle_missing:
                # Handle missing values based on data type
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0])
                logger.debug("Handled missing values")
            
            if remove_duplicates:
                initial_rows = len(df)
                df.drop_duplicates(inplace=True)
                dropped_rows = initial_rows - len(df)
                logger.debug(f"Removed {dropped_rows} duplicate rows")
            
            if handle_outliers:
                # Handle outliers using IQR method for numerical columns
                for col in df.select_dtypes(include=['int64', 'float64']):
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df[col] = df[col].clip(lower_bound, upper_bound)
                logger.debug("Handled outliers using IQR method")
            
            logger.info("Data cleaning completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise

    def extract_features(self, data: pd.DataFrame,
                        time_based: bool = True,
                        statistical: bool = True) -> pd.DataFrame:
        """
        Extract features from the dataset.
        
        Args:
            data: Input DataFrame
            time_based: Whether to extract time-based features
            statistical: Whether to extract statistical features
            
        Returns:
            pd.DataFrame: Data with extracted features
        """
        try:
            df = data.copy()
            
            # Extract time-based features if timestamp column exists
            if time_based and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
                logger.debug("Extracted time-based features")
            
            # Extract statistical features for numerical columns
            if statistical:
                numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
                
                # Rolling statistics
                window_sizes = [5, 10, 20]
                for col in numerical_cols:
                    for window in window_sizes:
                        df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                        df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                
                # Fill NaN values created by rolling calculations
                df.fillna(method='bfill', inplace=True)
                logger.debug("Extracted statistical features")
            
            logger.info("Feature extraction completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    def scale_features(self, data: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      scaler_type: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler or MinMaxScaler.
        
        Args:
            data: Input DataFrame
            columns: List of columns to scale (if None, all numerical columns are scaled)
            scaler_type: Type of scaler ('standard' or 'minmax')
            
        Returns:
            pd.DataFrame: Scaled data
        """
        try:
            df = data.copy()
            
            if columns is None:
                columns = df.select_dtypes(include=['int64', 'float64']).columns
            
            if scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError("Invalid scaler_type. Use 'standard' or 'minmax'")
            
            df[columns] = self.scaler.fit_transform(df[columns])
            
            # Save the scaler for future use
            scaler_path = os.path.join(MODELS_DIR, f'{scaler_type}_scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            logger.debug(f"Saved scaler to {scaler_path}")
            
            logger.info("Feature scaling completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise

    def prepare_training_data(self, data: pd.DataFrame,
                            target_column: str,
                            test_size: float = 0.2,
                            random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model training by splitting into training and testing sets.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple containing (X_train, X_test, y_train, y_test)
        """
        try:
            from sklearn.model_selection import train_test_split
            
            # Separate features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            logger.info("Data preparation for training completed successfully")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise

    def save_preprocessed_data(self, data: pd.DataFrame, filename: str) -> None:
        """
        Save preprocessed data to file.
        
        Args:
            data: Preprocessed DataFrame
            filename: Name of the file to save
        """
        try:
            # Create preprocessed data directory if it doesn't exist
            preprocessed_dir = os.path.join(DATA_DIR, 'preprocessed')
            os.makedirs(preprocessed_dir, exist_ok=True)
            
            # Add timestamp to filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = os.path.join(
                preprocessed_dir,
                f'{os.path.splitext(filename)[0]}_{timestamp}.parquet'
            )
            
            # Save as parquet for efficiency
            data.to_parquet(file_path, index=False)
            logger.info(f"Saved preprocessed data to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving preprocessed data: {str(e)}")
            raise

# Export the class
__all__ = ['DataPreprocessor']
