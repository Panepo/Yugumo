from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    StoppingCriteria,
)
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

class StopSequenceCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever a sequence of tokens is encountered.

    Args:
        stop_sequences (`str` or `List[str]`):
            The sequence (or list of sequences) on which to stop execution.
        tokenizer:
            The tokenizer used to decode the model outputs.
    """

    def __init__(self, stop_sequences, tokenizer):
        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        decoded_output = self.tokenizer.decode(input_ids.tolist()[0])
        return any(
            decoded_output.endswith(stop_sequence)
            for stop_sequence in self.stop_sequences
        )

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, load_in_4bit=bit4, torch_dtype="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048)
hf_llm = HuggingFacePipeline(pipeline=pipe)

hf_llm.pipeline._forward_params["stopping_criteria"] = StoppingCriteriaList(
    [StopSequenceCriteria(stop_tokens, tokenizer)]
)
