import os
import logging
from typing import Dict, Any

from utils.data import get_data
from utils.model import LLMGAN

logger = logging.getLogger(__name__)

def synthesize(
    general_cfg: Dict[str, Any],
    data_cfg:    Dict[str, Any],
    generation_cfg: Dict[str, Any]
) -> None:

    data_cfg["seed"] = general_cfg.get("seed")
    data_cfg["operation"] = general_cfg.get("operation")
    datasets = get_data(args=data_cfg)
    train_ds = datasets.get("train")

    logger.info(f"Retrieved training dataset: {train_ds}")
    if train_ds is None:
        raise ValueError("get_data did not return a 'train' split.")

    discriminator_template = load_template(data_cfg.get("discriminator_prompt_template_path"))
    
    llm_gan = LLMGAN(
        generator=general_cfg.get("generator_name_or_path"),
        discriminator=general_cfg.get("discriminator_name_or_path"),
        dis_prompt=discriminator_template
    )

    llm_gan.generate(
        data=train_ds, 
        gen_generation_params={},
        **generation_cfg
    )


def load_template(template_path: str) -> str:
    """Load prompt template from file."""
    if not template_path or not os.path.exists(template_path):
        logger.warning(f"Template file not found: {template_path}, using default")
        return "{prompt}"
    
    logger.info(f"Loading template from {template_path}")
    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        logger.info(f"Loaded discriminator template")
        return content