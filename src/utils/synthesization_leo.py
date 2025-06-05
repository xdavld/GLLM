import os
import torch
import pandas as pd
import logging
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, Dict, List, Tuple

from utils.data import get_data

logger = logging.getLogger(__name__)

def synthesize_leo(
    general_cfg: Dict[str, Any],
    data_cfg:    Dict[str, Any],
    training_cfg: Dict[str, Any]
) -> pd.DataFrame:
    """
    Implements a GAN-like system with:
    - Generator: Llama-3.1-8B-Instruct (generates synthetic data)
    - Discriminator: Llama-3.2-1B-Instruct (ranks/scores data)
    
    Returns a pandas DataFrame with columns: id, prompt, synthetic_data, discriminator_score
    """
    # 1. Obtain all prompt texts and real data via get_data
    data_cfg["seed"] = general_cfg.get("seed", 42)
    data_cfg["operation"] = general_cfg.get("operation")
    datasets = get_data(args=data_cfg)
    train_ds = datasets.get("train")
    logger.info(f"Retrieved training dataset: {train_ds}")
    if train_ds is None:
        raise ValueError("get_data did not return a 'train' split.")
    
    # Collect all prompt strings
    if hasattr(train_ds, 'column_names') and 'prompt' in train_ds.column_names:
        prompts = train_ds["prompt"]
    else:
        raise ValueError("Dataset must have a 'prompt' column")
    
    # Get real data for comparison (if available)
    real_data = []
    if hasattr(train_ds, 'column_names') and 'text' in train_ds.column_names:
        real_data = train_ds["text"]
    elif hasattr(train_ds, 'column_names') and 'response' in train_ds.column_names:
        real_data = train_ds["response"]
    
    # 2. Load generator model (Llama-3.1-8B)
    generator_path = general_cfg.get("generator_model_path")
    if not generator_path:
        raise ValueError("`generator_model_path` must be set in General section.")
    
    logger.info(f"Loading generator model from: {generator_path}")
    generator_tokenizer, generator_model = load_model(generator_path)
    
    # 3. Load discriminator model (Llama-3.2-1B)
    discriminator_path = general_cfg.get("discriminator_model_path")
    if not discriminator_path:
        raise ValueError("`discriminator_model_path` must be set in General section.")
    
    logger.info(f"Loading discriminator model from: {discriminator_path}")
    discriminator_tokenizer, discriminator_model = load_model(discriminator_path)
    
    # 4. Load prompt templates
    generator_template = load_template(data_cfg.get("generator_prompt_template_path"))
    discriminator_template = load_template(data_cfg.get("discriminator_prompt_template_path"))
    
    # 5. Generation parameters
    max_new_tokens = training_cfg.get("max_new_tokens", 512)
    temperature = training_cfg.get("temperature", 0.8)
    top_k = training_cfg.get("top_k", 50)
    top_p = training_cfg.get("top_p", 0.9)
    do_sample = training_cfg.get("do_sample", True)
    repetition_penalty = training_cfg.get("repetition_penalty", 1.1)
    num_return_sequences = training_cfg.get("num_return_sequences", 1)
    
    # Apply sample_size limit if specified
    sample_size = data_cfg.get("sample_size")
    if sample_size and sample_size > 0:
        prompts = prompts[:sample_size]
        if real_data:
            real_data = real_data[:sample_size]
        logger.info(f"Limited to {sample_size} samples")
    
    results = []
    
    # 6. Main generation and discrimination loop
    for idx, prompt in enumerate(prompts):
        logger.info(f"Processing prompt {idx+1}/{len(prompts)}...")
        
        # Generate synthetic data using generator
        synthetic_texts = generate_synthetic_data(
            prompt=prompt,
            template=generator_template,
            tokenizer=generator_tokenizer,
            model=generator_model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences
        )
        
        # Prepare data for discrimination (synthetic + real data sample)
        discrimination_data = []
        
        # Add synthetic data
        for syn_text in synthetic_texts:
            discrimination_data.append({
                "text": syn_text,
                "type": "synthetic",
                "prompt": prompt
            })
        
        # Add some real data for comparison (if available)
        if real_data and len(real_data) > idx:
            discrimination_data.append({
                "text": real_data[idx],
                "type": "real",
                "prompt": prompt
            })
        
        # Score data using discriminator
        scores = discriminate_data(
            data=discrimination_data,
            template=discriminator_template,
            tokenizer=discriminator_tokenizer,
            model=discriminator_model
        )

        ### Leo update weights discriminator
        
        # Store results
        for i, (data_item, score) in enumerate(zip(discrimination_data, scores)):
            results.append({
                'prompt': prompt,
                'text': data_item["text"],
                'type': data_item["type"],
                'discriminator_score': score,
                'prompt_idx': idx
            })
    
    # 7. Create DataFrame
    df = pd.DataFrame(results)
    df['id'] = range(len(df))
    df = df[['id', 'prompt', 'text', 'type', 'discriminator_score', 'prompt_idx']]
    
    # 8. Save results
    output_path = data_cfg.get("output_path", "synthetic_data.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Generated {len(df[df['type'] == 'synthetic'])} synthetic samples")
    logger.info(f"Processed {len(df[df['type'] == 'real'])} real samples")
    
    synthetic_avg = df[df['type'] == 'synthetic']['discriminator_score'].mean()
    logger.info(f"Average discriminator score for synthetic: {synthetic_avg:.2f}")
    
    if len(df[df['type'] == 'real']) > 0:
        real_avg = df[df['type'] == 'real']['discriminator_score'].mean()
        logger.info(f"Average discriminator score for real: {real_avg:.2f}")
    
    return df


def load_model(model_path: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load tokenizer and model from the given path."""
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model path '{model_path}' is not a directory.")
    if not os.path.exists(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(f"Model not found at {model_path}.")
    
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    return tokenizer, model


def load_template(template_path: str) -> str:
    """Load prompt template from file."""
    if not template_path or not os.path.exists(template_path):
        logger.warning(f"Template file not found: {template_path}, using default")
        return "{prompt}"
    
    logger.info(f"Loading template from {template_path}")
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def generate_synthetic_data(
    prompt: str,
    template: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    **generation_kwargs
) -> List[str]:
    """Generate synthetic data using the generator model."""
    formatted_prompt = format_prompt_with_template(prompt, template, tokenizer)
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048  # Add max length to prevent issues
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=generation_kwargs.get("max_new_tokens", 512),
            do_sample=generation_kwargs.get("do_sample", True),
            top_k=generation_kwargs.get("top_k", 50),
            top_p=generation_kwargs.get("top_p", 0.9),
            temperature=generation_kwargs.get("temperature", 0.8),
            repetition_penalty=generation_kwargs.get("repetition_penalty", 1.1),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=generation_kwargs.get("num_return_sequences", 1)
        )
    
    # Decode generated tokens
    generated_texts = []
    for output in outputs:
        new_tokens = output[inputs['input_ids'].shape[-1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        generated_texts.append(text)
    
    return generated_texts


def discriminate_data(
    data: List[Dict],
    template: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM
) -> List[int]:
    """Score data using the discriminator model."""
    # Prepare input for discriminator
    input_array = [{"text": item["text"]} for item in data]
    input_json = json.dumps(input_array, indent=2)
    
    # Format discriminator prompt
    discriminator_prompt = template.replace("{{input}}", input_json)
    formatted_prompt = format_prompt_with_template(discriminator_prompt, "{prompt}", tokenizer)
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048  # Add max length to prevent issues
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,  # Use greedy decoding for consistency
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    new_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    # Parse JSON response
    try:
        scores_data = json.loads(response)
        scores = [item["score"] for item in scores_data]
        
        # Ensure we have the right number of scores
        if len(scores) != len(data):
            logger.warning(f"Expected {len(data)} scores, got {len(scores)}. Using fallback.")
            scores = [5] * len(data)  # Fallback to neutral scores
            
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse discriminator response: {e}. Using fallback scores.")
        logger.warning(f"Raw response: {response}")
        scores = [5] * len(data)  # Fallback to neutral scores
    
    return scores


def format_prompt_with_template(prompt: str, template: str, tokenizer: AutoTokenizer) -> str:
    """Format prompt using template and chat template if available."""
    formatted_content = template.replace("{prompt}", prompt)
    
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": formatted_content}
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        return formatted_content