"""
Testing module for the Intrusion and Anomaly Detection System (IADS).
Implements comprehensive testing for all system components.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Tuple, Any
import tempfile

from src.core.config import logger, config, DATA_DIR
from src.preprocessing.data_preprocessing import DataPreprocessor
from src.detection.rule_based_detection import RuleBasedDetector, DetectionResult
from src.model.ai_model import AIModel
from src.model.model_optimization import ModelOptimizer
from src.core.db import DatabaseManager, SystemStatus, DetectionEvents

class TestDataPreprocessing(unittest.TestCase):
    """Test cases for data preprocessing functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.preprocessor = DataPreprocessor()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
            'bytes_transferred': np.random.randint(1000, 10000, 100),
            'source_ip': ['192.168.1.' + str(i % 255) for i in range(100)],
            'destination_port': np.random.randint(1, 65535, 100),
            'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], 100)
        })
        
        # Add some anomalies
        self.sample_data.loc[10:15, 'bytes_transferred'] = 999999  # Anomalous traffic
        
        # Save sample data
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data_path = os.path.join(self.temp_dir, 'sample_data.csv')
        self.sample_data.to_csv(self.sample_data_path, index=False)

    def test_load_data(self):
        """Test data loading functionality."""
        try:
            loaded_data = self.preprocessor.load_data(self.sample_data_path)
            self.assertEqual(len(loaded_data), len(self.sample_data))
            self.assertTrue(all(col in loaded_data.columns 
                              for col in self.sample_data.columns))
        except Exception as e:
            self.fail(f"Data loading failed: {str(e)}")

    def test_clean_data(self):
        """Test data cleaning functionality."""
        # Add some missing values and duplicates
        dirty_data = self.sample_data.copy()
        dirty_data.loc[0:5, 'bytes_transferred'] = np.nan
        dirty_data = pd.concat([dirty_data, dirty_data.head()])
        
        cleaned_data = self.preprocessor.clean_data(dirty_data)
        
        self.assertEqual(len(cleaned_data), len(dirty_data) - 5)  # Check duplicates removed
        self.assertEqual(cleaned_data.isna().sum().sum(), 0)  # Check no missing values

    def test_feature_extraction(self):
        """Test feature extraction functionality."""
        features = self.preprocessor.extract_features(self.sample_data)
        
        # Check time-based features
        self.assertTrue('hour' in features.columns)
        self.assertTrue('day_of_week' in features.columns)
        
        # Check statistical features
        self.assertTrue(any('rolling_mean' in col for col in features.columns))
        self.assertTrue(any('rolling_std' in col for col in features.columns))

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)

class TestRuleBasedDetection(unittest.TestCase):
    """Test cases for rule-based detection functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.detector = RuleBasedDetector()
        
        # Create sample data with known anomalies
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
            'bytes_transferred': np.random.randint(1000, 10000, 100),
            'login_success': [True] * 95 + [False] * 5,  # 5 failed logins
            'source_ip': ['192.168.1.10'] * 95 + ['192.168.1.20'] * 5,
            'destination_port': [80] * 90 + [12345] * 10,  # 10 unusual ports
            'direction': ['inbound'] * 90 + ['outbound'] * 10
        })
        
        # Add some anomalies
        self.sample_data.loc[90:, 'bytes_transferred'] = 999999  # Large transfers

    def test_high_traffic_detection(self):
        """Test high traffic volume detection."""
        results = self.detector.detect_high_traffic(self.sample_data)
        self.assertTrue(len(results) > 0)
        self.assertTrue(all(isinstance(r, DetectionResult) for r in results))
        self.assertTrue(any(r.severity == "medium" for r in results))

    def test_failed_login_detection(self):
        """Test failed login detection."""
        results = self.detector.detect_failed_logins(self.sample_data)
        self.assertTrue(len(results) > 0)
        self.assertTrue(any(r.source_ip == "192.168.1.20" for r in results))

    def test_unusual_port_detection(self):
        """Test unusual port detection."""
        results = self.detector.detect_unusual_ports(self.sample_data)
        self.assertTrue(len(results) > 0)
        self.assertTrue(any(r.description.find("12345") != -1 for r in results))

class TestAIModel(unittest.TestCase):
    """Test cases for AI model functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.input_shape = 10
        self.num_classes = 2
        self.model = AIModel(self.input_shape, self.num_classes)
        
        # Create sample training data
        self.X_train = np.random.random((100, self.input_shape))
        self.y_train = np.random.randint(0, self.num_classes, 100)
        self.y_train = tf.keras.utils.to_categorical(self.y_train, self.num_classes)
        
        # Create validation data
        self.X_val = np.random.random((20, self.input_shape))
        self.y_val = np.random.randint(0, self.num_classes, 20)
        self.y_val = tf.keras.utils.to_categorical(self.y_val, self.num_classes)

    def test_model_building(self):
        """Test model building functionality."""
        self.model.build_model()
        self.assertIsNotNone(self.model.model)
        self.assertEqual(self.model.model.input_shape[1], self.input_shape)
        self.assertEqual(self.model.model.output_shape[1], self.num_classes)

    def test_model_training(self):
        """Test model training functionality."""
        self.model.build_model()
        self.model.train_model(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            model_name="test_model"
        )
        self.assertIsNotNone(self.model.history)
        
        # Check if model files were created
        model_path = os.path.join(MODELS_DIR, "test_model.h5")
        self.assertTrue(os.path.exists(model_path))

    def test_model_prediction(self):
        """Test model prediction functionality."""
        self.model.build_model()
        self.model.train_model(
            self.X_train, self.y_train,
            self.X_val, self.y_val
        )
        
        predictions = self.model.predict(self.X_val)
        self.assertEqual(predictions.shape[0], len(self.X_val))
        self.assertEqual(predictions.shape[1], self.num_classes)

    def tearDown(self):
        """Clean up test environment."""
        # Remove test model files
        for f in os.listdir(MODELS_DIR):
            if f.startswith("test_model"):
                os.remove(os.path.join(MODELS_DIR, f))

class TestModelOptimization(unittest.TestCase):
    """Test cases for model optimization functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.optimizer = ModelOptimizer()
        
        # Create and train a simple model for testing
        self.input_shape = 10
        self.num_classes = 2
        self.model = AIModel(self.input_shape, self.num_classes)
        self.model.build_model()
        
        # Sample data for calibration
        self.calibration_data = np.random.random((10, self.input_shape))

    def test_quantization(self):
        """Test model quantization functionality."""
        quantized_model_path = self.optimizer.quantize_model(
            self.model.model,
            self.calibration_data,
            quantization_type='float16'
        )
        self.assertTrue(os.path.exists(quantized_model_path))
        self.assertTrue(os.path.getsize(quantized_model_path) > 0)

    def test_optimization_for_inference(self):
        """Test model optimization for inference."""
        optimized_model = self.optimizer.optimize_for_inference(
            self.model.model,
            (None, self.input_shape)
        )
        self.assertIsNotNone(optimized_model)

    def test_benchmark(self):
        """Test model benchmarking functionality."""
        results = self.optimizer.benchmark_model(
            self.model.model,
            self.calibration_data
        )
        self.assertTrue('avg_inference_time' in results)
        self.assertTrue('memory_usage' in results)

def run_all_tests():
    """Run all test cases and generate report."""
    try:
        # Create test suite
        suite = unittest.TestSuite()
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDataPreprocessing))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRuleBasedDetection))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAIModel))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestModelOptimization))
        
        # Run tests and capture results
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Generate test report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        }
        
        # Save report
        report_dir = os.path.join(DATA_DIR, 'test_reports')
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(
            report_dir,
            f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info(f"Test report generated: {report_path}")
        logger.info(f"Success rate: {report['success_rate']:.2f}%")
        
        return report
        
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        raise

if __name__ == "__main__":
    run_all_tests()
