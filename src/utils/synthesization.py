import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, Dict, List

from utils.data import get_data

def synthesize(
    general_cfg: Dict[str, Any],
    data_cfg:    Dict[str, Any],
    training_cfg: Dict[str, Any]
) -> List[str]:
    """
    Loads Llama-3.1-8B-Instruct model via AutoTokenizer/AutoModelForCausalLM,
    then generates synthetic data using each prompt obtained from get_data().

    Returns a list of generated outputs, one per prompt.
    """
    # 1. Obtain all prompt texts via get_data
    data_cfg["seed"] = general_cfg.get("seed")
    data_cfg["operation"] = general_cfg.get("operation")
    datasets = get_data(args=data_cfg)

    # 'train' split should contain a column named "prompt"
    prompts_list = datasets.get("train")
    if prompts_list is None:
        raise ValueError("get_data did not return a 'train' split.")

    # Extract the actual list of prompt strings
    if hasattr(prompts_list, "column_names") and "prompt" in prompts_list.column_names:
        # HF Dataset: prompts_list["prompt"] is a list of strings
        prompts_list = prompts_list["prompt"]
    elif isinstance(prompts_list, list) and len(prompts_list) > 0:
        # If get_data returned a plain list of strings
        pass  # prompts_list is already a list
    else:
        raise ValueError("Could not extract prompt strings from get_data output.")

    # 2. Get model_path from general_cfg
    model_path = general_cfg.get("model_path")
    if not model_path:
        raise ValueError("`model_path` must be set in General section.")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model path '{model_path}' is not a directory.")
    if not os.path.exists(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please run `snapshot_download('meta-llama/Llama-3.1-8B-Instruct', cache_dir=<that path>)` first."
        )
    print(f"Using local model from {model_path}")

    # 3. Load tokenizer and model using Auto classes
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    # 4. Generation parameters
    max_new_tokens      = training_cfg.get("max_new_tokens")
    temperature         = training_cfg.get("temperature")
    top_k               = training_cfg.get("top_k")
    top_p               = training_cfg.get("top_p")
    do_sample           = training_cfg.get("do_sample")
    repetition_penalty  = training_cfg.get("repetition_penalty")
    num_return_sequences = training_cfg.get("num_return_sequences")

    results = []

    # 5. Loop through each prompt and generate
    for idx, prompt in enumerate(prompts):
        print(f"Processing prompt {idx}...")
        formatted_prompt = format_prompt(prompt, tokenizer)
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        print(f"Generating synthetic data for prompt {idx}...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=num_return_sequences
            )

        # Decode generated tokens
        gen_texts_for_prompt = []
        for output in outputs:
            new_tokens = output[inputs['input_ids'].shape[-1]:]
            txt = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            gen_texts_for_prompt.append(txt)

        # If only one, append that string, else join them
        if len(gen_texts_for_prompt) == 1:
            results.append(gen_texts_for_prompt[0])
        else:
            results.append(gen_texts_for_prompt)

    return results


def format_prompt(prompt: str, tokenizer) -> str:
    """
    Format prompt using Llama's chat template if available.
    """
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates high-quality synthetic data."},
            {"role": "user",   "content": prompt}
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )