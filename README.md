# Entwurf und Implementierung eines LLM-basierten synthetischen Frameworks für eine datengetriebene Produktinnovation am Beispiel der Bierherstellung

Finetuning:

export TOKENIZERS_PARALLELISM=false

accelerate launch --config_file configs/deepspeed_zero3.yaml main.py configs/config_fine_tuning.yaml