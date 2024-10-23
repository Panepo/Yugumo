from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os

load_dotenv()
backend = os.getenv("BACKEND")

if backend == "cuda":
  device = "cuda"
elif backend == "cpu":
  device = "cpu"
else:
  raise ValueError(f"Unknown backend: {backend}")

stop_tokens = ["Observation:"]
model_path = "./models/hfmodels/Mistral-7B-Instruct-v0.3/"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
