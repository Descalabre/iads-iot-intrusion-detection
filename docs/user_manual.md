# User Manual: Intrusion and Anomaly Detection System (IADS) for IoT Devices

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Configuration](#configuration)
5. [Running the System](#running-the-system)
6. [Training the AI Model](#training-the-ai-model)
7. [Model Optimization](#model-optimization)
8. [Testing](#testing)
9. [Web Dashboard](#web-dashboard)
10. [Troubleshooting](#troubleshooting)
11. [FAQs](#faqs)
12. [Support](#support)

## Overview

The Intrusion and Anomaly Detection System (IADS) is a hybrid system that combines rule-based detection with AI models to identify intrusions and anomalies in IoT environments. The system is optimized for deployment on resource-constrained devices such as Raspberry Pi.

### Key Features
- Hybrid detection approach (rule-based + AI)
- Real-time monitoring and alerts
- Web-based dashboard
- Model optimization for edge devices
- Comprehensive testing framework
- Detailed logging and reporting

## Installation

1. Ensure Python 3.7+ is installed:
   ```bash
   python3 --version
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/iads.git
   cd iads
   ```

3. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
iads/
├── src/
│   ├── core/              # Core functionality
│   │   ├── config.py      # Configuration management
│   │   └── db.py         # Database operations
│   ├── preprocessing/     # Data preprocessing
│   ├── detection/        # Detection algorithms
│   ├── model/           # AI model implementation
│   ├── testing/         # Test suite
│   └── ui/             # Web dashboard
├── data/               # Data directory
├── models/            # Saved models
└── docs/             # Documentation
```

## Configuration

### System Configuration

1. Create a configuration file:
   ```bash
   cp config.example.json config.json
   ```

2. Edit `config.json` to adjust:
   - Database settings
   - Model parameters
   - Detection thresholds
   - Logging levels

Example configuration:
```json
{
    "database": {
        "host": "localhost",
        "port": 3306,
        "database": "iads_db",
        "user": "iads_user",
        "password": "your_password"
    },
    "model": {
        "type": "dense",
        "input_shape": 10,
        "num_classes": 2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50
    },
    "detection": {
        "traffic_volume_threshold": 1000,
        "failed_login_threshold": 5,
        "unusual_port_threshold": 0.01
    }
}
```

### Data Directory Setup

1. Create required directories:
   ```bash
   mkdir -p data/{raw,processed,results}
   mkdir -p models/{saved,optimized}
   ```

2. Place your data files in the appropriate directories:
   - Raw data: `data/raw/`
   - Processed data: `data/processed/`
   - Results: `data/results/`

## Running the System

### Basic Usage

1. Run detection on new data:
   ```bash
   python src/main.py --mode run --data data/raw/sample.csv
   ```

2. View real-time results:
   ```bash
   python src/ui/ui_dashboard.py
   ```
   Access dashboard at `http://localhost:5000`

### Advanced Options

```bash
python src/main.py --mode run \
    --data data/raw/sample.csv \
    --model-name my_model \
    --config custom_config.json
```

Available arguments:
- `--mode`: Operation mode (`run`, `train`, `test`, `optimize`)
- `--data`: Path to input data
- `--model-name`: Name of model to use/save
- `--config`: Custom configuration file

## Training the AI Model

### Preparing Training Data

1. Format your training data as CSV with columns:
   - timestamp
   - feature_1, feature_2, ...
   - label

2. Split your data:
   ```bash
   python src/preprocessing/split_data.py \
       --input data/raw/training.csv \
       --train-ratio 0.8
   ```

### Training Process

1. Basic training:
   ```bash
   python src/main.py --mode train \
       --data data/processed/train.csv \
       --model-name my_model
   ```

2. Advanced training options:
   ```bash
   python src/main.py --mode train \
       --data data/processed/train.csv \
       --model-name my_model \
       --epochs 100 \
       --batch-size 64
   ```

### Monitoring Training

1. View training progress in terminal
2. Access TensorBoard:
   ```bash
   tensorboard --logdir models/logs
   ```

## Model Optimization

### Quantization

```bash
python src/main.py --mode optimize \
    --model-name my_model \
    --quantization-type float16
```

Supported quantization types:
- `float16`: Reduced precision
- `dynamic`: Dynamic range quantization
- `int8`: 8-bit integer quantization

### Pruning

```bash
python src/main.py --mode optimize \
    --model-name my_model \
    --target-sparsity 0.5
```

### Benchmarking

```bash
python src/model/benchmark.py --model-name my_model
```

## Testing

### Running Tests

1. Run all tests:
   ```bash
   python src/main.py --mode test
   ```

2. Run specific test suite:
   ```bash
   python -m unittest src/testing/testing.py -k TestDataPreprocessing
   ```

### Test Reports

- Test reports are saved in `data/test_reports/`
- View detailed results in the dashboard

## Web Dashboard

### Features

1. Real-time monitoring:
   - Detection counts
   - System status
   - CPU/Memory usage

2. Historical data:
   - Detection trends
   - Performance metrics
   - System logs

### Customization

Edit `src/ui/templates/dashboard.html` to:
- Modify layout
- Add new visualizations
- Update styling

## Troubleshooting

### Common Issues

1. Database Connection:
   ```
   Error: Cannot connect to database
   Solution: Check database credentials in config.json
   ```

2. Model Loading:
   ```
   Error: Model file not found
   Solution: Verify model path in configuration
   ```

3. Memory Issues:
   ```
   Error: Out of memory
   Solution: Reduce batch size or optimize model
   ```

### Logging

- Logs are stored in `logs/iads.log`
- Set log level in configuration:
  ```json
  {
      "log_level": "DEBUG"
  }
  ```

### Debug Mode

Run with debug logging:
```bash
python src/main.py --mode run --debug
```

## FAQs

### General Questions

**Q: Can I run this on devices other than Raspberry Pi?**  
A: Yes, but optimization is targeted for Raspberry Pi. For other devices, adjust the optimization parameters in `config.json`.

**Q: How do I update detection rules?**  
A: Edit `src/detection/rule_based_detection.py`. Each rule is a class method that can be modified or extended.

**Q: How often should I retrain the model?**  
A: Retrain when:
- New types of attacks are identified
- False positive/negative rates increase
- System environment changes significantly

### Technical Questions

**Q: What's the minimum hardware requirement?**  
A: 
- CPU: 1GHz+
- RAM: 512MB+
- Storage: 1GB+

**Q: Can I use custom neural network architectures?**  
A: Yes, extend the `ModelArchitecture` class in `src/model/ai_model.py`.

**Q: How to handle different data formats?**  
A: Add custom data loaders in `src/preprocessing/data_preprocessing.py`.

## Support

### Getting Help

1. Check the documentation in `docs/`
2. Review logs in `logs/iads.log`
3. Submit issues on GitHub
4. Contact maintainers:
   - Email: support@iads.com
   - Discord: IADS Community Server

### Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

For detailed contribution guidelines, see `CONTRIBUTING.md`.
