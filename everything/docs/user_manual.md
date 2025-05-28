# User Manual: Intrusion and Anomaly Detection System (IADS) for IoT Devices

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Configuration](#configuration)
5. [Running the System](#running-the-system)
6. [Training the AI Model](#training-the-ai-model)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)
9. [FAQs](#faqs)
10. [Support](#support)

## Overview

The Intrusion and Anomaly Detection System (IADS) is a hybrid system that combines rule-based detection with AI models to identify intrusions and anomalies in IoT environments. The system is optimized for deployment on resource-constrained devices such as Raspberry Pi.

### Key Features
- Hybrid detection approach (rule-based + AI)
- Real-time monitoring and alerts
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
│   ├── core/              # Core functionality (configuration, logging)
│   ├── preprocessing/     # Data preprocessing
│   ├── model/             # AI model implementation
│   ├── testing/           # Test suite
│   └── main.py            # Main entry point for training and testing
├── data/                  # Data directory
├── models/                # Saved models
└── docs/                  # Documentation
```

## Configuration

### System Configuration

1. Create a configuration file:
   ```bash
   cp config.example.json config.json
   ```

2. Edit `config.json` to adjust:
   - Model parameters
   - Detection thresholds
   - Logging levels

Example configuration:
```json
{
    "model": {
        "type": "dense",
        "input_shape": 10,
        "num_classes": 2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50
    },
    "detection": {
        "threshold": 0.8,
        "window_size": 100
    },
    "logging": {
        "log_level": "DEBUG"
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

1. Train the AI model:
   ```bash
   python src/main.py --mode train --data data/processed/train.csv --model-name my_model
   ```

2. Start the API server:
   ```bash
   # Development mode
   python -m src.api.app

   # Or with uvicorn directly
   uvicorn src.api.app:app --host 0.0.0.0 --port 5000 --reload
   ```

3. Access the web dashboard:
   - Open http://localhost:5000 in your browser
   - Use the interface to make predictions and view results

### API Endpoints

- `GET /`: Web dashboard interface
- `GET /health`: Health check endpoint
- `POST /predict`: Make predictions
  ```json
  {
    "features": [0.1, 0.2, 0.3, ...]
  }
  ```
- `GET /model/info`: Get model information

### Deployment Options

1. Local Development:
   ```bash
   uvicorn src.api.app:app --host 0.0.0.0 --port 5000 --reload
   ```

2. Production with Gunicorn:
   ```bash
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.app:app
   ```

3. Cloud Deployment:
   - VPS Setup:
     1. Install dependencies: `pip install -r requirements.txt`
     2. Configure Nginx as reverse proxy
     3. Set up SSL with Let's Encrypt
     4. Run with Gunicorn

   - Serverless:
     - Package the application for AWS Lambda
     - Configure API Gateway
     - Set environment variables for model paths

### Available Arguments

- `--mode`: Operation mode (`train`, `test`)
- `--data`: Path to input data (required for training)
- `--model-name`: Name of model to use/save
- `--config`: Custom configuration file
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--host`: API server host (default: localhost)
- `--port`: API server port (default: 5000)

## Training the AI Model

### Preparing Training Data

1. Format your training data as CSV with columns:
   - timestamp
   - feature_1, feature_2, ...
   - label

2. Split your data:
   ```bash
   python src/preprocessing/split_data.py --input data/raw/training.csv --train-ratio 0.8
   ```

### Training Process

1. Basic training:
   ```bash
   python src/main.py --mode train --data data/processed/train.csv --model-name my_model
   ```

2. Advanced training options:
   ```bash
   python src/main.py --mode train --data data/processed/train.csv --model-name my_model --epochs 100 --batch-size 64
   ```

### Monitoring Training

1. View training progress in terminal

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

## Troubleshooting

### Common Issues

1. Model Loading:
   ```
   Error: Model file not found
   Solution: Verify model path in configuration
   ```

2. Memory Issues:
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
python src/main.py --mode train --debug
```

## FAQs

### General Questions

**Q: Can I run this on devices other than Raspberry Pi?**  
A: Yes, but optimization is targeted for Raspberry Pi. For other devices, adjust the optimization parameters in `config.json`.

**Q: How often should I retrain the model?**  
A: Retrain when:
- New types of attacks are identified
- False positive/negative rates increase
- System environment changes significantly

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
