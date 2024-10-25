from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from dotenv import load_dotenv
import os

load_dotenv()
backend = os.getenv("BACKEND")
bit4 = True if os.getenv("BIT4") == 'true' else False

if backend == "cuda":
  device = "cuda"
elif backend == "cpu":
  device = "cpu"
else:
  raise ValueError(f"Unknown backend: {backend}")

stop_tokens = ["Observation:"]
model_path = "./models/hfmodels/Mistral-7B-Instruct-v0.3/"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, load_in_4bit=bit4, torch_dtype="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200, device=device, stop_token=stop_tokens)
hf_llm = HuggingFacePipeline(pipeline=pipe)
