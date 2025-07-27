# LLM-Based Synthetic Framework for Data-Driven Product Innovation in Beer Production

This repository contains the implementation of a Large Language Model (LLM) based synthetic framework designed for data-driven product innovation, demonstrated through beer production optimization.

## Overview

The framework consists of several key components:
- **Generator Fine-tuning**: Training the LLM to understand data structure and patterns
- **GLLM Training**: Core language model training pipeline
- **Data Synthesis**: Synthetic data generation capabilities
- **Evaluation**: Performance assessment and validation tools
- **Sensory Prediction**: Beer sensory profile prediction using VAE models

## Quick Start

### Prerequisites
- Python 3.8+
- Required dependencies (see `requirements.txt`)
- Accelerate and DeepSpeed for distributed training

### Installation

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

## Usage

### 1. Generator Fine-tuning

Fine-tune the generator to learn data structure and develop an understanding of the dataset patterns:

```bash
export TOKENIZERS_PARALLELISM=false
accelerate launch --config_file configs/deepspeed_zero3.yaml main.py configs/config_fine_tuning.yaml
```

### 2. General Training, Synthesis, and Evaluation

All main operations (GLLM training, data synthesis, and evaluation) are executed from the `src` directory:

1. Navigate to the source directory:
   ```bash
   cd src
   ```

2. Run the desired configuration:
   ```bash
   python main.py configs/DATASET/CONFIG_FILE
   ```

   **Example for beer dataset:**
   ```bash
   python main.py configs/beer/config_fine_tuning.yaml
   ```

### 3. Configuration Management

- Configuration files are located in the `configs/` directory
- Adjust file paths and hyperparameters in the respective config files as needed
- Each dataset has its own configuration subdirectory (e.g., `configs/beer/`)

### 4. Beer Sensory Profile Prediction

For the beer project, you can predict sensory target profiles using the following two-step process:

#### Step 1: Train the VAE Model
```bash
python model_training.py
```

#### Step 2: Run Inference
```bash
python inference.py
```

## Project Structure

```
├── src/                    # Source code directory
├── configs/               # Configuration files
│   ├── beer/             # Beer-specific configurations
│   └── deepspeed_zero3.yaml
├── model_training.py     # VAE model training script
├── inference.py          # Prediction inference script
└── main.py              # Main execution script
```

## Configuration Files

The framework uses YAML configuration files to manage:
- Model hyperparameters
- Data paths and preprocessing settings
- Training configurations
- Evaluation metrics

Modify the configuration files in the `configs/` directory to customize the framework for your specific use case.

## Contributing

Please ensure all contributions follow the established code structure and include appropriate configuration files for new datasets or model variants.

## License

[Add your license information here]
