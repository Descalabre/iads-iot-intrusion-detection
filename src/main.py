"""
Main entry point for the Intrusion and Anomaly Detection System (IADS).
Handles command-line interface and orchestrates system components.
"""

import argparse
import sys
import os
from datetime import datetime
import json
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.core.config import logger, config, DATA_DIR, MODELS_DIR
from src.preprocessing.data_preprocessing import DataPreprocessor
from src.detection.rule_based_detection import RuleBasedDetector
from src.model.ai_model import AIModel
from src.model.model_optimization import ModelOptimizer
from src.testing.testing import run_all_tests
from src.core.db import DatabaseManager, SystemStatus, DetectionEvents

class IADSController:
    """Main controller class for the IADS system."""
    
    def __init__(self):
        """Initialize system components."""
        self.preprocessor = DataPreprocessor()
        self.rule_detector = RuleBasedDetector()
        self.model_optimizer = ModelOptimizer()
        
        # Initialize AI model with config
        model_config = config.get('model', {})
        self.ai_model = AIModel(
            input_shape=model_config.get('input_shape', 10),
            num_classes=model_config.get('num_classes', 2),
            model_type=model_config.get('type', 'dense')
        )

    def train(self, data_path: str, **kwargs) -> None:
        """
        Train the AI model with provided data.
        
        Args:
            data_path: Path to training data
            **kwargs: Additional training parameters
        """
        try:
            logger.info("Starting training workflow...")
            
            # Load and preprocess data
            raw_data = self.preprocessor.load_data(data_path)
            cleaned_data = self.preprocessor.clean_data(raw_data)
            processed_data = self.preprocessor.extract_features(cleaned_data)
            
            # Prepare training data
            target_column = kwargs.get('target_column', 'label')
            X_train, X_test, y_train, y_test = self.preprocessor.prepare_training_data(
                processed_data,
                target_column=target_column,
                test_size=kwargs.get('test_size', 0.2)
            )
            
            # Build and train model
            self.ai_model.build_model()
            
            model_name = kwargs.get('model_name', f"iads_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.ai_model.train_model(
                X_train, y_train,
                X_test, y_test,
                model_name=model_name
            )
            
            # Evaluate model
            metrics = self.ai_model.evaluate_model(X_test, y_test)
            
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

    def run_detection(self, data_path: str, **kwargs) -> Dict[str, Any]:
        """
        Run the detection system on provided data.
        
        Args:
            data_path: Path to data for analysis
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing detection results
        """
        try:
            logger.info("Starting detection workflow...")
            
            # Load and preprocess data
            raw_data = self.preprocessor.load_data(data_path)
            cleaned_data = self.preprocessor.clean_data(raw_data)
            processed_data = self.preprocessor.extract_features(cleaned_data)
            
            # Rule-based detection
            rule_results = self.rule_detector.analyze_data(processed_data)
            
            # AI model detection
            if kwargs.get('use_ai', True):
                model_name = kwargs.get('model_name')
                if model_name:
                    self.ai_model.load_model(model_name)
                
                if not hasattr(self.ai_model, 'model') or self.ai_model.model is None:
                    self.ai_model.build_model()
                
                # Prepare data for AI model
                X = processed_data[self.ai_model.feature_columns] if hasattr(self.ai_model, 'feature_columns') else processed_data
                ai_predictions = self.ai_model.predict(X)
            
            # Combine results
            results = {
                'timestamp': datetime.now().isoformat(),
                'rule_based_detections': [r.__dict__ for r in rule_results],
                'ai_detections': ai_predictions.tolist() if kwargs.get('use_ai', True) else None,
                'parameters': kwargs
            }
            
            # Save results
            results_dir = os.path.join(DATA_DIR, 'detection_results')
            os.makedirs(results_dir, exist_ok=True)
            
            results_path = os.path.join(
                results_dir,
                f"detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            # Update system status
            SystemStatus.insert_status(
                detections=len(rule_results),
                cpu_usage=os.getloadavg()[0],
                uptime=str(datetime.now()),
                memory_usage=0.0  # TODO: Implement memory usage monitoring
            )
            
            logger.info(f"Detection completed. Results saved to {results_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error in detection workflow: {str(e)}")
            raise

    def optimize(self, model_name: str, **kwargs) -> None:
        """
        Optimize a trained model for deployment.
        
        Args:
            model_name: Name of the model to optimize
            **kwargs: Additional optimization parameters
        """
        try:
            logger.info(f"Starting model optimization for {model_name}...")
            
            # Load the model
            self.ai_model.load_model(model_name)
            
            # Quantization
            if kwargs.get('quantize', True):
                quantized_model = self.model_optimizer.quantize_model(
                    self.ai_model.model,
                    quantization_type=kwargs.get('quantization_type', 'float16')
                )
                logger.info(f"Model quantized: {quantized_model}")
            
            # Pruning
            if kwargs.get('prune', False):
                pruned_model = self.model_optimizer.prune_model(
                    self.ai_model.model,
                    target_sparsity=kwargs.get('target_sparsity', 0.5)
                )
                logger.info("Model pruned successfully")
            
            # Optimize for inference
            if kwargs.get('optimize_inference', True):
                optimized_model = self.model_optimizer.optimize_for_inference(
                    self.ai_model.model,
                    self.ai_model.input_shape
                )
                logger.info("Model optimized for inference")
            
            # Benchmark
            if kwargs.get('benchmark', True):
                benchmark_results = self.model_optimizer.benchmark_model(
                    self.ai_model.model,
                    np.random.random((100, *self.ai_model.input_shape))
                )
                logger.info("Benchmark completed")
            
        except Exception as e:
            logger.error(f"Error in optimization workflow: {str(e)}")
            raise

def main():
    """Main entry point for the IADS system."""
    parser = argparse.ArgumentParser(description="Intrusion and Anomaly Detection System (IADS)")
    
    parser.add_argument(
        "--mode",
        type=str,
        default="run",
        choices=["run", "train", "test", "optimize"],
        help="Mode to run the system"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        help="Path to input data file"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        help="Name of the model to use/save"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file"
    )
    
    # Training specific arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )
    
    # Optimization specific arguments
    parser.add_argument(
        "--quantization-type",
        type=str,
        choices=['float16', 'dynamic', 'int8'],
        default='float16',
        help="Type of quantization to apply"
    )
    
    parser.add_argument(
        "--target-sparsity",
        type=float,
        default=0.5,
        help="Target sparsity for model pruning"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize controller
        controller = IADSController()
        
        if args.mode == "train":
            if not args.data:
                raise ValueError("Data file path must be provided in training mode!")
            
            controller.train(
                args.data,
                model_name=args.model_name,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
        
        elif args.mode == "run":
            if not args.data:
                raise ValueError("Data file path must be provided in run mode!")
            
            results = controller.run_detection(
                args.data,
                model_name=args.model_name,
                use_ai=True
            )
            
            # Print summary of results
            print("\nDetection Results Summary:")
            print(f"Rule-based detections: {len(results['rule_based_detections'])}")
            if results['ai_detections']:
                print(f"AI model detections: {len(results['ai_detections'])}")
        
        elif args.mode == "test":
            test_report = run_all_tests()
            print("\nTest Results Summary:")
            print(f"Total tests: {test_report['total_tests']}")
            print(f"Success rate: {test_report['success_rate']:.2f}%")
        
        elif args.mode == "optimize":
            if not args.model_name:
                raise ValueError("Model name must be provided in optimize mode!")
            
            controller.optimize(
                args.model_name,
                quantization_type=args.quantization_type,
                target_sparsity=args.target_sparsity
            )
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
