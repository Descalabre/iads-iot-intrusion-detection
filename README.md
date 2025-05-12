# Intrusion and Anomaly Detection System (IADS) for IoT Devices

## Overview

IADS is a comprehensive security solution that combines rule-based detection with AI models to identify intrusions and anomalies in IoT environments. The system is specifically optimized for deployment on resource-constrained devices such as Raspberry Pi.

### Key Features

- **Hybrid Detection Approach**
  - Rule-based detection for known patterns
  - AI-powered detection for anomalies
  - Real-time monitoring and alerts

- **Optimized for Edge Devices**
  - Model quantization and pruning
  - Efficient inference
  - Resource usage monitoring

- **Modern Web Dashboard**
  - Real-time monitoring
  - Interactive visualizations
  - System health metrics

- **Comprehensive Testing**
  - Automated test suite
  - Performance benchmarking
  - Detailed reporting

## Project Structure

```
iads/
├── src/                    # Source code
│   ├── core/              # Core functionality
│   │   ├── config.py      # Configuration management
│   │   └── db.py         # Database operations
│   ├── preprocessing/     # Data preprocessing
│   │   └── data_preprocessing.py
│   ├── detection/        # Detection algorithms
│   │   └── rule_based_detection.py
│   ├── model/           # AI model implementation
│   │   ├── ai_model.py
│   │   └── model_optimization.py
│   ├── testing/         # Test suite
│   │   └── testing.py
│   ├── ui/             # Web dashboard
│   │   ├── ui_dashboard.py
│   │   └── templates/
│   └── main.py         # Main entry point
├── data/               # Data directory
│   ├── raw/           # Raw input data
│   ├── processed/     # Processed data
│   └── results/       # Detection results
├── models/            # Model storage
│   ├── saved/        # Saved models
│   └── optimized/    # Optimized models
├── docs/             # Documentation
│   ├── user_manual.md
│   └── architecture_paper.md
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Quick Start

1. **Installation**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Basic Usage**
   ```bash
   # Run detection
   python src/main.py --mode run --data data/raw/sample.csv

   # Train model
   python src/main.py --mode train --data data/raw/training.csv

   # Launch dashboard
   python src/ui/ui_dashboard.py
   ```

3. **View Results**
   - Access dashboard at `http://localhost:5000`
   - Check detection results in `data/results/`
   - View logs in `logs/iads.log`

## Documentation

- [User Manual](docs/user_manual.md) - Detailed usage instructions
- [Architecture Paper](docs/architecture_paper.md) - System design and implementation details

## Requirements

- Python 3.7+
- TensorFlow 2.x
- MySQL/MariaDB
- Additional dependencies in requirements.txt

### Hardware Requirements

- CPU: 1GHz+
- RAM: 512MB+
- Storage: 1GB+
- Network connectivity for dashboard

## Development

### Running Tests

```bash
# Run all tests
python src/main.py --mode test

# Run specific test suite
python -m unittest src/testing/testing.py -k TestDataPreprocessing
```

### Model Optimization

```bash
# Optimize model for edge deployment
python src/main.py --mode optimize \
    --model-name my_model \
    --quantization-type float16
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Documentation: Check the [User Manual](docs/user_manual.md)
- Issues: Submit via GitHub Issues
- Community: Join our Discord server
- Email: support@iads.com

## Acknowledgments

- TensorFlow team for the optimization toolkit
- scikit-learn community for preprocessing tools
- Flask team for the web framework

## Citation

If you use this software in your research, please cite:

```bibtex
@software{iads2023,
  title = {Intrusion and Anomaly Detection System},
  author = {IADS Team},
  year = {2023},
  url = {https://github.com/your-repo/iads}
}
