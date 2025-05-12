"""
AI Model module for the Intrusion and Anomaly Detection System (IADS).
Implements deep learning models for anomaly and intrusion detection.
"""

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, BatchNormalization,
    Input, Conv1D, MaxPooling1D, Flatten
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import os
import json
from datetime import datetime
import joblib

from src.core.config import logger, config, MODELS_DIR

class ModelArchitecture:
    """Defines different model architectures available in the system."""
    
    @staticmethod
    def dense_network(input_shape: int, num_classes: int) -> Model:
        """
        Create a dense neural network architecture.
        
        Args:
            input_shape: Number of input features
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        inputs = Input(shape=(input_shape,))
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        return Model(inputs=inputs, outputs=outputs)

    @staticmethod
    def lstm_network(input_shape: Tuple[int, int], num_classes: int) -> Model:
        """
        Create an LSTM network architecture.
        
        Args:
            input_shape: (sequence_length, num_features)
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        inputs = Input(shape=input_shape)
        x = LSTM(64, return_sequences=True)(inputs)
        x = Dropout(0.3)(x)
        x = LSTM(32)(x)
        x = Dropout(0.2)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        return Model(inputs=inputs, outputs=outputs)

    @staticmethod
    def cnn_network(input_shape: Tuple[int, int], num_classes: int) -> Model:
        """
        Create a CNN architecture.
        
        Args:
            input_shape: (sequence_length, num_features)
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        inputs = Input(shape=input_shape)
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        return Model(inputs=inputs, outputs=outputs)

class AIModel:
    """Main AI model class for anomaly and intrusion detection."""
    
    def __init__(self, 
                 input_shape: Union[int, Tuple[int, int]],
                 num_classes: int,
                 model_type: str = 'dense'):
        """
        Initialize the AI model.
        
        Args:
            input_shape: Shape of input data
            num_classes: Number of output classes
            model_type: Type of model architecture ('dense', 'lstm', or 'cnn')
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_type = model_type
        self.model: Optional[Model] = None
        self.history: Optional[Dict] = None
        
        # Load model configuration
        self.config = config.get('model', {})
        
        # Create models directory if it doesn't exist
        os.makedirs(MODELS_DIR, exist_ok=True)

    def build_model(self) -> None:
        """Build the neural network model based on specified architecture."""
        try:
            if self.model_type == 'dense':
                self.model = ModelArchitecture.dense_network(self.input_shape, self.num_classes)
            elif self.model_type == 'lstm':
                self.model = ModelArchitecture.lstm_network(self.input_shape, self.num_classes)
            elif self.model_type == 'cnn':
                self.model = ModelArchitecture.cnn_network(self.input_shape, self.num_classes)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Compile model
            self.model.compile(
                optimizer=Adam(learning_rate=self.config.get('learning_rate', 0.001)),
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            
            logger.info(f"Built {self.model_type} model successfully")
            self.model.summary()
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise

    def _create_callbacks(self, model_name: str) -> List[tf.keras.callbacks.Callback]:
        """
        Create training callbacks.
        
        Args:
            model_name: Name of the model for saving checkpoints
            
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.get('early_stopping_patience', 10),
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=os.path.join(MODELS_DIR, f'{model_name}_checkpoint.h5'),
                monitor='val_loss',
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            ),
            TensorBoard(
                log_dir=os.path.join(MODELS_DIR, 'logs', model_name),
                histogram_freq=1
            )
        ]
        return callbacks

    def train_model(self,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val: np.ndarray,
                   y_val: np.ndarray,
                   model_name: str = None) -> None:
        """
        Train the model with the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_name: Name for saving the model
        """
        try:
            if self.model is None:
                raise ValueError("Model not built. Call build_model() first.")
            
            # Generate model name if not provided
            if model_name is None:
                model_name = f"iads_{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create callbacks
            callbacks = self._create_callbacks(model_name)
            
            # Train the model
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.get('epochs', 50),
                batch_size=self.config.get('batch_size', 32),
                callbacks=callbacks,
                verbose=1
            )
            
            # Save the final model
            self.save_model(model_name)
            
            # Save training history
            self._save_history(model_name)
            
            logger.info(f"Model training completed successfully: {model_name}")
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Model predictions
        """
        try:
            if self.model is None:
                raise ValueError("Model not built or loaded. Call build_model() or load_model() first.")
            
            predictions = self.model.predict(X)
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            if self.model is None:
                raise ValueError("Model not built or loaded. Call build_model() or load_model() first.")
            
            results = self.model.evaluate(X_test, y_test, verbose=0)
            metrics = dict(zip(self.model.metrics_names, results))
            
            logger.info("Model evaluation completed:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

    def save_model(self, model_name: str) -> None:
        """
        Save the trained model and its configuration.
        
        Args:
            model_name: Name of the model
        """
        try:
            if self.model is None:
                raise ValueError("No model to save. Call build_model() first.")
            
            # Save model
            model_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
            self.model.save(model_path)
            
            # Save model configuration
            config_path = os.path.join(MODELS_DIR, f"{model_name}_config.json")
            model_config = {
                'input_shape': self.input_shape,
                'num_classes': self.num_classes,
                'model_type': self.model_type,
                'config': self.config
            }
            with open(config_path, 'w') as f:
                json.dump(model_config, f)
            
            logger.info(f"Model saved successfully: {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_name: str) -> None:
        """
        Load a saved model and its configuration.
        
        Args:
            model_name: Name of the model to load
        """
        try:
            # Load model configuration
            config_path = os.path.join(MODELS_DIR, f"{model_name}_config.json")
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            
            # Update instance variables
            self.input_shape = model_config['input_shape']
            self.num_classes = model_config['num_classes']
            self.model_type = model_config['model_type']
            self.config.update(model_config['config'])
            
            # Load model
            model_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
            self.model = tf.keras.models.load_model(model_path)
            
            logger.info(f"Model loaded successfully: {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _save_history(self, model_name: str) -> None:
        """
        Save training history to file.
        
        Args:
            model_name: Name of the model
        """
        try:
            if self.history is None:
                logger.warning("No training history to save")
                return
            
            history_path = os.path.join(MODELS_DIR, f"{model_name}_history.json")
            with open(history_path, 'w') as f:
                json.dump(self.history.history, f)
            
            logger.debug(f"Training history saved: {history_path}")
            
        except Exception as e:
            logger.error(f"Error saving training history: {str(e)}")
            # Don't raise here as this is not critical

# Export the classes
__all__ = ['AIModel', 'ModelArchitecture']
