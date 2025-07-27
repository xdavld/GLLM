# Entwurf und Implementierung eines LLM-basierten synthetischen Frameworks für eine datengetriebene Produktinnovation am Beispiel der Bierherstellung

Finetuning des generators, dass er die son struktur lernt und ein gefühl für die daten bekommt:

export TOKENIZERS_PARALLELISM=false

accelerate launch --config_file configs/deepspeed_zero3.yaml main.py configs/config_fine_tuning.yaml

Sonst gibt es noch training des GLLM, die synthetisierung der daten und evaluation alle ruft man auf mit.
1. in src ordner gehen
2. python main.py configs/DATASET/CONFIG_FILE
z.B. python main.py configs/beer/config_fine_tuning.yaml

in den einzelnen configs files dann ggf. dann die pfade anpassen oder auch hyperparameter für die gewünschte aktion.

dann für das bierprojekt noch die prediction für das szensorische zielbild, was mit folgendem befehl ausgeführt werden kann:
zuerst den vae trainieren:
python model_training.py

dann predicten:
python inference.py