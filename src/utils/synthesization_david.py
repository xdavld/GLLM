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
    pass