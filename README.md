# GLLM: A GAN-Inspired LLM Framework for the Synthetic Generation of Tabular Datasets

## Description
Generative framework that integrates Generative Adversarial Network (GAN) principles with Large Language Models (LLMs) to synthetically augment small-scale tabular dataset

## Fine-Tuning the Generator

Set up and run the following to fine-tune the generator model so it learns the data structure and gains familiarity with the dataset:

```bash
export TOKENIZERS_PARALLELISM=false
accelerate launch --config_file configs/deepspeed_zero3.yaml main.py configs/<DATASET>/config_fine_tuning.yaml
```

## GLLM Training, Data Synthesis, and Evaluation

Use this workflow to train the generalized LLM (GLLM), generate synthetic data, and evaluate model performance:

1. Navigate to the `src` folder:

   ```bash
   cd src
   ```
2. Adjust file paths or hyperparameters in the config files as needed.
3. Run the main script with the desired config file:

   ```bash
   python main.py configs/<DATASET>/<CONFIG_FILE>
   ```

   For example:

   ```bash
   python main.py configs/beer/config_fine_tuning.yaml
   ```

## Sensory Target Prediction (Beer Project)

Predict sensory target profiles using a Conditional Variational Autoencoder (CVAE):

1. Navigate to the `prediction` folder:

   ```bash
   cd src/prediction
   ```

2. Train the CVAE model:

   ```bash
   python model_training.py
   ```
3. Generate predictions:

   ```bash
   python inference.py
   ```
