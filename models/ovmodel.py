from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    StoppingCriteria,
)

device = "GPU"
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}
stop_tokens = ["Observation:"]
model_path = "ovmodels/Mistral-7B-Instruct-v0.3-ov-int4"


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


ov_llm = HuggingFacePipeline.from_model_id(
    model_id=model_path,
    task="text-generation",
    backend="openvino",
    model_kwargs={
        "device": device,
        "ov_config": ov_config,
        "trust_remote_code": True,
    },
    pipeline_kwargs={"max_new_tokens": 2048},
)
ov_llm = ov_llm.bind(skip_prompt=True, stop=["Observation:"])

tokenizer = ov_llm.pipeline.tokenizer
ov_llm.pipeline._forward_params["stopping_criteria"] = StoppingCriteriaList(
    [StopSequenceCriteria(stop_tokens, tokenizer)]
)
