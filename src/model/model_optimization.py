"""
Model Optimization module for the Intrusion and Anomaly Detection System (IADS).
Implements various optimization techniques for deploying models on resource-constrained devices.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
import os
import json
from datetime import datetime
import tempfile

from src.core.config import logger, config, MODELS_DIR

class ModelOptimizer:
    """Handles various model optimization techniques for deployment."""
    
    def __init__(self):
        """Initialize the model optimizer with configuration."""
        self.config = config.get('optimization', {})
        self.optimization_dir = os.path.join(MODELS_DIR, 'optimized')
        os.makedirs(self.optimization_dir, exist_ok=True)

    def quantize_model(self,
                      model: tf.keras.Model,
                      calibration_data: Optional[np.ndarray] = None,
                      quantization_type: str = 'float16') -> str:
        """
        Quantize the model to reduce its size and improve inference speed.
        
        Args:
            model: Original Keras model
            calibration_data: Representative dataset for quantization
            quantization_type: Type of quantization ('float16', 'dynamic', or 'int8')
            
        Returns:
            Path to the optimized model
        """
        try:
            # Generate output filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(
                self.optimization_dir,
                f"quantized_model_{quantization_type}_{timestamp}.tflite"
            )
            
            # Create TFLite converter
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            if quantization_type == 'float16':
                logger.info("Applying float16 quantization")
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            
            elif quantization_type == 'dynamic':
                logger.info("Applying dynamic range quantization")
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            elif quantization_type == 'int8':
                if calibration_data is None:
                    raise ValueError("Calibration data required for int8 quantization")
                
                logger.info("Applying int8 quantization")
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                
                def representative_dataset():
                    for data in calibration_data:
                        yield [np.expand_dims(data, axis=0).astype(np.float32)]
                
                converter.representative_dataset = representative_dataset
            
            else:
                raise ValueError(f"Unsupported quantization type: {quantization_type}")
            
            # Convert the model
            tflite_model = converter.convert()
            
            # Save the quantized model
            with open(output_file, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"Model quantized successfully: {output_file}")
            
            # Save metadata
            metadata = {
                'original_model_size': os.path.getsize(model.name),
                'quantized_model_size': os.path.getsize(output_file),
                'quantization_type': quantization_type,
                'timestamp': timestamp
            }
            
            metadata_file = os.path.join(
                self.optimization_dir,
                f"quantization_metadata_{timestamp}.json"
            )
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error during model quantization: {str(e)}")
            raise

    def prune_model(self,
                   model: tf.keras.Model,
                   validation_data: Tuple[np.ndarray, np.ndarray],
                   target_sparsity: float = 0.5) -> tf.keras.Model:
        """
        Apply weight pruning to reduce model size while maintaining accuracy.
        
        Args:
            model: Original Keras model
            validation_data: Tuple of (x_val, y_val) for evaluating pruning
            target_sparsity: Target sparsity ratio (0.0 to 1.0)
            
        Returns:
            Pruned Keras model
        """
        try:
            import tensorflow_model_optimization as tfmot
            
            # Define pruning schedule
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.0,
                    final_sparsity=target_sparsity,
                    begin_step=0,
                    end_step=1000
                )
            }
            
            # Apply pruning to the model
            pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
            
            # Compile the pruned model
            pruned_model.compile(
                optimizer=model.optimizer,
                loss=model.loss,
                metrics=model.metrics
            )
            
            # Fine-tune the pruned model
            callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
            
            pruned_model.fit(
                validation_data[0],
                validation_data[1],
                epochs=self.config.get('pruning_epochs', 5),
                batch_size=32,
                callbacks=callbacks,
                validation_split=0.1
            )
            
            # Strip pruning wrapper
            final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
            
            # Save the pruned model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(
                self.optimization_dir,
                f"pruned_model_{timestamp}.h5"
            )
            final_model.save(output_file)
            
            logger.info(f"Model pruned successfully: {output_file}")
            return final_model
            
        except Exception as e:
            logger.error(f"Error during model pruning: {str(e)}")
            raise

    def optimize_for_inference(self,
                             model: tf.keras.Model,
                             input_shape: Tuple[int, ...]) -> tf.keras.Model:
        """
        Optimize model for inference by fusing operations and converting variables to constants.
        
        Args:
            model: Original Keras model
            input_shape: Shape of input tensor
            
        Returns:
            Optimized Keras model
        """
        try:
            # Convert Keras model to SavedModel format
            with tempfile.TemporaryDirectory() as temp_dir:
                tf.saved_model.save(model, temp_dir)
                
                # Convert SavedModel to GraphDef
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(
                    tf.io.read_file(
                        os.path.join(temp_dir, 'saved_model.pb')
                    ).numpy()
                )
            
            # Optimize the graph
            from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
            
            @tf.function
            def serving_fn(x):
                return model(x)
            
            serving_fn = serving_fn.get_concrete_function(
                tf.TensorSpec(input_shape, tf.float32)
            )
            
            frozen_func = convert_variables_to_constants_v2(serving_fn)
            frozen_func.graph.as_graph_def()
            
            # Convert back to Keras model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(
                self.optimization_dir,
                f"inference_optimized_{timestamp}"
            )
            
            tf.saved_model.save(
                tf.keras.models.Model(
                    inputs=frozen_func.inputs,
                    outputs=frozen_func.outputs
                ),
                output_file
            )
            
            optimized_model = tf.keras.models.load_model(output_file)
            
            logger.info(f"Model optimized for inference: {output_file}")
            return optimized_model
            
        except Exception as e:
            logger.error(f"Error during inference optimization: {str(e)}")
            raise

    def benchmark_model(self,
                      model: tf.keras.Model,
                      input_data: np.ndarray,
                      num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark model performance (inference time, memory usage).
        
        Args:
            model: Model to benchmark
            input_data: Sample input data
            num_runs: Number of inference runs for averaging
            
        Returns:
            Dictionary containing benchmark results
        """
        try:
            import time
            import psutil
            
            # Warm-up run
            model.predict(input_data[:1])
            
            # Measure inference time
            inference_times = []
            for _ in range(num_runs):
                start_time = time.time()
                model.predict(input_data[:1])
                inference_times.append(time.time() - start_time)
            
            # Measure memory usage
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss
            model.predict(input_data[:1])
            memory_after = process.memory_info().rss
            
            results = {
                'avg_inference_time': np.mean(inference_times),
                'std_inference_time': np.std(inference_times),
                'memory_usage': (memory_after - memory_before) / 1024 / 1024,  # MB
                'model_size': os.path.getsize(model.name) / 1024 / 1024  # MB
            }
            
            logger.info("Benchmark results:")
            for metric, value in results.items():
                logger.info(f"{metric}: {value:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during model benchmarking: {str(e)}")
            raise

    def export_for_edge(self,
                       model: tf.keras.Model,
                       target_platform: str = 'tflite',
                       optimization_level: str = 'balanced') -> str:
        """
        Export model for edge deployment with platform-specific optimizations.
        
        Args:
            model: Original Keras model
            target_platform: Target platform ('tflite', 'edge_tpu', 'tensorrt')
            optimization_level: Optimization level ('minimal', 'balanced', 'aggressive')
            
        Returns:
            Path to the exported model
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if target_platform == 'tflite':
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                
                if optimization_level == 'minimal':
                    converter.optimizations = []
                elif optimization_level == 'balanced':
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                elif optimization_level == 'aggressive':
                    converter.optimizations = [
                        tf.lite.Optimize.DEFAULT,
                        tf.lite.Optimize.EXPERIMENTAL_SPARSITY
                    ]
                
                tflite_model = converter.convert()
                output_file = os.path.join(
                    self.optimization_dir,
                    f"edge_model_{target_platform}_{optimization_level}_{timestamp}.tflite"
                )
                
                with open(output_file, 'wb') as f:
                    f.write(tflite_model)
            
            elif target_platform == 'edge_tpu':
                # Add Edge TPU-specific optimizations
                raise NotImplementedError("Edge TPU optimization not implemented yet")
            
            elif target_platform == 'tensorrt':
                # Add TensorRT optimization
                raise NotImplementedError("TensorRT optimization not implemented yet")
            
            else:
                raise ValueError(f"Unsupported target platform: {target_platform}")
            
            logger.info(f"Model exported for edge deployment: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error during edge export: {str(e)}")
            raise

# Export the class
__all__ = ['ModelOptimizer']
