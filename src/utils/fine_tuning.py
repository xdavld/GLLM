import os
import glob
import shutil
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data import get_data

def fine_tune(general_cfg, data_cfg, training_cfg):
    """
    Fine-tunes the given model using the SFTTrainer.

    This function assumes you have an SFT trainer available that integrates with Hugging Face's 
    Trainer API. It requires model, tokenizer, and a training dataset.

    Args:
        args: The arguments containing model name, training data, and other configurations.

    Returns:
        The fine-tuned model.
    """

    data_cfg["seed"] = general_cfg.get("seed", None)
    data_cfg["operation"] = general_cfg.get("operation", None)
    datasets = get_data(args=data_cfg)

    model_name_or_path = general_cfg.get("model_name_or_path", None)

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if "learning_rate" in training_cfg:
        lr = training_cfg["learning_rate"]
        training_cfg["learning_rate"] = float(lr)

    training_args = SFTConfig(
        **training_cfg,
        max_length=tokenizer.model_max_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"] if datasets["eval"] else None,
        processing_class=tokenizer,
    )

    trainer.train()