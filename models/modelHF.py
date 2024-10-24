from langchain_core.language_models import BaseLanguageModel
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

class LLM(BaseLanguageModel):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

hf_llm = LLM(model, tokenizer)
