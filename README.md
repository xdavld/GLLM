# LLM-Based Synthetic Framework for Data-Driven Product Innovation

## Description

Design and implementation of an LLM-based synthetic framework for data-driven product innovation, demonstrated through beer production.

## Fine-Tuning the Generator

Set up and run the following to fine-tune the generator model so it learns the data structure and gains familiarity with the dataset:

```bash
export TOKENIZERS_PARALLELISM=false
accelerate launch --config_file configs/deepspeed_zero3.yaml main.py configs/config_fine_tuning.yaml
```

## GLLM Training, Data Synthesis, and Evaluation

Use this workflow to train the generalized LLM (GLLM), generate synthetic data, and evaluate model performance:

1. Navigate to the `src` folder:

   ```bash
   cd src
   ```
2. Run the main script with the desired config file:

   ```bash
   python main.py configs/<DATASET>/config_file.yaml
   ```

   For example:

   ```bash
   python main.py configs/beer/config_fine_tuning.yaml
   ```
3. Adjust file paths or hyperparameters in the config files as needed.

## Sensory Target Prediction (Beer Project)

Predict sensory target profiles using a Conditional Variational Autoencoder (CVAE):

1. Train the CVAE model:

   ```bash
   python model_training.py
   ```
2. Generate predictions:

   ```bash
   python inference.py
   ```
