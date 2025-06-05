import os
import torch
import pandas as pd
import logging
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, Dict, List, Tuple

from utils.data import get_data

logger = logging.getLogger(__name__)

def synthesize(
    general_cfg: Dict[str, Any],
    data_cfg:    Dict[str, Any],
    training_cfg: Dict[str, Any]
) -> pd.DataFrame:
    """
    Implements a GAN-like system with:
    - Generator: Llama-3.1-8B-Instruct (generates synthetic data)
    - Discriminator: Llama-3.2-1B-Instruct (classifies data)
    
    Returns a pandas DataFrame with columns: prompt, generated_data, label, discriminator_pred
    """
    # Obtain all prompt texts and real data via get_data
    data_cfg["seed"] = general_cfg.get("seed")
    data_cfg["operation"] = general_cfg.get("operation")
    datasets = get_data(args=data_cfg)
    train_ds = datasets.get("train")
    logger.info(f"Retrieved training dataset: {train_ds}")
    if train_ds is None:
        raise ValueError("get_data did not return a 'train' split.")
    
    # Extract the actual list of prompt strings
    if hasattr(train_ds, "column_names") and "prompt" in train_ds.column_names:
        prompts = train_ds["prompt"]
    else:
        raise ValueError("Could not extract prompt strings from get_data output.")
    
    # Get real data for comparison 
    real_data = []
    if hasattr(train_ds, 'column_names') and 'input_data' in train_ds.column_names:
        logger.info("Extracting real data from input_data column")
        for input_json_str in train_ds["input_data"]:
            try:
                examples = json.loads(input_json_str)
                # Format each example as a JSON array string to match synthetic data format
                for example in examples:
                    formatted_example = json.dumps([example])
                    real_data.append(formatted_example)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse input_data JSON: {e}")
        logger.info(f"Extracted {len(real_data)} real examples from input_data")

    # Load generator model (Llama-3.1-8B)
    generator_path = general_cfg.get("generator_model_path")
    if not generator_path:
        raise ValueError("`generator_model_path` must be set in General section.")
    
    logger.info(f"Loading generator model from: {generator_path}")
    generator_tokenizer, generator_model = load_model(generator_path)
    
    # Load discriminator model (Llama-3.2-1B)
    discriminator_path = general_cfg.get("discriminator_model_path")
    if not discriminator_path:
        raise ValueError("`discriminator_model_path` must be set in General section.")
    
    logger.info(f"Loading discriminator model from: {discriminator_path}")
    discriminator_tokenizer, discriminator_model = load_model(discriminator_path)
    
    # Load discriminator template
    discriminator_template = load_template(data_cfg.get("discriminator_prompt_template_path"))
    
    # Generation parameters
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
    
    # Main generation and discrimination loop
    for idx, prompt in enumerate(prompts):
        logger.info(f"Processing prompt {idx+1}/{len(prompts)}...")
        logger.info(f"Generator input prompt: {prompt[:200]}...")  # Show first 200 chars
        
        # Generate synthetic data using generator
        synthetic_texts = generate_synthetic_data(
            prompt=prompt,
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
        
        # Process synthetic data
        for syn_idx, syn_text in enumerate(synthetic_texts):
            logger.info(f"Generator output {syn_idx+1}: {syn_text[:200]}...")  # Show first 200 chars
            
            # Create discrimination data for synthetic sample
            discrimination_data = [{"text": syn_text}]
            logger.info(f"Discriminator input (synthetic): {json.dumps(discrimination_data, indent=2)[:300]}...")
            
            # Get discriminator prediction for synthetic data
            discriminator_pred = discriminate_data(
                data=discrimination_data,
                template=discriminator_template,
                tokenizer=discriminator_tokenizer,
                model=discriminator_model
            )
                        
            # Store synthetic data result
            results.append({
                'prompt': prompt,
                'generated_data': syn_text,
                'label': 0,  # 0 for synthetic (ground truth)
                'discriminator_pred': discriminator_pred
            })
        
        # Process real data if available
        if real_data and len(real_data) > idx:
            real_text = real_data[idx]
            logger.info(f"Real data input: {real_text[:200]}...")  # Show first 200 chars
            
            # Create discrimination data for real sample
            discrimination_data = [{"text": real_text}]
            logger.info(f"Discriminator input (real): {json.dumps(discrimination_data, indent=2)[:300]}...")
            
            # Get discriminator prediction for real data
            discriminator_pred = discriminate_data(
                data=discrimination_data,
                template=discriminator_template,
                tokenizer=discriminator_tokenizer,
                model=discriminator_model
            )
                    
            # Store real data result
            results.append({
                'prompt': prompt,
                'generated_data': real_text,
                'label': 1,  # 1 for real (ground truth)
                'discriminator_pred': discriminator_pred
            })
    
    # Create DataFrame (without id column)
    df = pd.DataFrame(results)
    df = df[['prompt', 'generated_data', 'label', 'discriminator_pred']]
    
    # Save results
    output_path = data_cfg.get("output_path")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Generated {len(df[df['label'] == 0])} synthetic samples")
    logger.info(f"Processed {len(df[df['label'] == 1])} real samples")
    
    # Calculate metrics for synthetic and real data separately
    synthetic_df = df[df['label'] == 0]
    real_df = df[df['label'] == 1]
    
    if len(synthetic_df) > 0:        
        # Accuracy for synthetic (should predict 0)
        synthetic_accuracy = (synthetic_df['discriminator_pred'] == 0).sum() / len(synthetic_df)
        logger.info(f"Discriminator accuracy on synthetic data: {synthetic_accuracy:.2f}")

    if len(real_df) > 0:
        # Accuracy for real (should predict 1)
        real_accuracy = (real_df['discriminator_pred'] == 1).sum() / len(real_df)
        logger.info(f"Discriminator accuracy on real data: {real_accuracy:.2f}")
    
    # Overall accuracy
    overall_accuracy = (df['discriminator_pred'] == df['label']).sum() / len(df)
    logger.info(f"Overall discriminator accuracy: {overall_accuracy:.2f}")
    
    return df

def discriminate_data(
    data: List[Dict],
    template: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM
) -> int:
    """Classify data using the discriminator model and return single prediction."""
    # Prepare input for discriminator
    input_array = [{"text": item["text"]} for item in data]
    input_json = json.dumps(input_array, indent=2)
    
    # Format discriminator prompt
    discriminator_prompt = template.replace("{{input}}", input_json)
    formatted_prompt = format_prompt_with_template(discriminator_prompt, "{prompt}", tokenizer)
    
    logger.info(f"Discriminator formatted prompt: {formatted_prompt[:500]}...")
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,    
            do_sample=False,      
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    new_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    logger.info(f"Discriminator raw response: '{response}'")

    return response

def generate_synthetic_data(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    **generation_kwargs
) -> List[str]:
    """Generate synthetic data using the generator model."""
    formatted_prompt = format_prompt(prompt, tokenizer)
    
    logger.info(f"Generator formatted prompt: {formatted_prompt[:500]}...")  # Show first 500 chars
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Only include generation parameters if do_sample=True
    generation_params = {
        "max_new_tokens": generation_kwargs.get("max_new_tokens", 512),
        "do_sample": generation_kwargs.get("do_sample", True),
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "num_return_sequences": generation_kwargs.get("num_return_sequences", 1)
    }
    
    # Add sampling parameters only if do_sample=True
    if generation_kwargs.get("do_sample", True):
        generation_params.update({
            "top_k": generation_kwargs.get("top_k", 50),
            "top_p": generation_kwargs.get("top_p", 0.9),
            "temperature": generation_kwargs.get("temperature", 0.8),
            "repetition_penalty": generation_kwargs.get("repetition_penalty", 1.1)
        })
    
    logger.info(f"Generation parameters: {generation_params}")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_params)
    
    # Decode generated tokens
    generated_texts = []
    for output in outputs:
        new_tokens = output[inputs['input_ids'].shape[-1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        generated_texts.append(text)
        logger.info(f"Generated text (decoded): {text[:300]}...")  # Show first 300 chars
    
    return generated_texts


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
        content = f.read().strip()
        logger.info(f"Loaded discriminator template: {content[:200]}...")
        return content


def format_prompt(prompt: str, tokenizer: AutoTokenizer) -> str:
    """Format prompt using Llama's chat template if available."""
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates high-quality synthetic data."},
            {"role": "user", "content": prompt}
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        return prompt


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