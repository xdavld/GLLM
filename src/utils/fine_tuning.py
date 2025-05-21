def fine_tune(args):
    """
    Fine-tunes the given model using the SFTTrainer.

    This function assumes you have an SFT trainer available that integrates with Hugging Face's 
    Trainer API. It requires model, tokenizer, and a training dataset.

    Args:
        args: The arguments containing model name, training data, and other configurations.

    Returns:
        The fine-tuned model.
    """
    from trl import SFTTrainer, SFTConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=12,
        save_steps=500,
        logging_steps=100,
        evaluation_strategy="epoch",
        eval_steps=500,
        save_total_limit=5,
        load_best_model_at_end=True,
        max_length=tokenizer.model_max_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    return trainer.model