import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

INPUT_DIR = cfg["INPUT_DIR"]
OUTPUT_DIR = cfg["OUTPUT_DIR"]
MODEL_NAME = cfg["MODEL_NAME"]
LANGUAGE = cfg["LANGUAGE"]
CHUNK_SEC = cfg["CHUNK_SEC"]
OVERLAP = 3.0
NUM_WORKERS = 1
DEVICE = cfg["DEVICE"]  # GPU(CUDA) 강제 사용


