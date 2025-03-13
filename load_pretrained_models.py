"""
Run this script once to pre-load the models into the cache.
It will download the models and store them in the cache directory.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from videollama2 import model_init

device = "cuda"  # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct", torch_dtype="auto", device_map="auto", cache_dir="./cache"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", cache_dir="./cache")

model_path = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"
model, processor, tokenizer = model_init(model_path, cache_dir="./cache")
