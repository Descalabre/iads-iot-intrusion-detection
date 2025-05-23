"""
Alternative main training script with dynamic model input shape.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
import json
import os

from src.core.config import logger, MODELS_DIR
from src.preprocessing.data_preprocessing import DataPreprocessor
from src.model.ai_model import AIModel

def train(data_path: str, **kwargs) -> None:
    """
    Train the AI model with provided data.
    
    Args:
        data_path: Path to training data
        **kwargs: Additional training parameters
    """
    try:
        logger.info("Starting training workflow...")
        
        # Load and preprocess data
        dp = DataPreprocessor()
        raw_data = dp.load_data(data_path)
        cleaned_data = dp.clean_data(raw_data)
        processed_data = dp.extract_features(cleaned_data)
        
        # Prepare training data
        target_column = kwargs.get('target_column', 'label')
        X_train, X_test, y_train, y_test = dp.prepare_training_data(
            processed_data,
            target_column=target_column,
            test_size=kwargs.get('test_size', 0.2)
        )
        
        # Build and train model with dynamic input shape
        ai_model = AIModel(input_shape=(X_train.shape[1],), num_classes=2, model_type="dense")
        ai_model.build_model()
        
        model_name = kwargs.get('model_name', f"iads_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        ai_model.train_model(
            X_train, y_train,
            X_test, y_test,
            model_name=model_name
        )
        
        # Evaluate model
        metrics = ai_model.evaluate_model(X_test, y_test)
        
        # Save training results
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'metrics': metrics,
            'parameters': kwargs
        }
        
        results_path = os.path.join(MODELS_DIR, f"{model_name}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Training completed. Results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Error in training workflow: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train IADS model with dynamic input shape")
    parser.add_argument("--data", type=str, required=True, help="Path to training data file")
    parser.add_argument("--model-name", type=str, default=None, help="Name of the model to save")
    parser.add_argument("--target-column", type=str, default="label", help="Target column name")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test data proportion")
    args = parser.parse_args()
    
    train(
        args.data,
        model_name=args.model_name,
        target_column=args.target_column,
        test_size=args.test_size
    )
