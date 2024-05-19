import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding

from . import model_base

# Mapping from path -> (tokenizer_path, model_path)
_PATH_REMAPS = {
    'CarperAI/openai_summarize_tldr_sft': ('EleutherAI/gpt-j-6b', 'CarperAI/openai_summarize_tldr_sft')
}


class HFModel(model_base.ModelBase):
    def __init__(self, model_name: str):
        tokenizer_path, model_path = _PATH_REMAPS.get(
            model_name, (model_name, model_name))
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path)

    def predict(self, inputs: str) -> str:
        inputs = self.tokenizer(inputs, return_tensors='pt')
        tokens = self.model.generate(**inputs)
        return self.tokenizer.batch_decode(tokens)

    def decode_step(self, inputs: BatchEncoding) -> str:
        inputs = self.tokenizer(inputs, return_tensors='pt')
        tokens = self.model.generate(
            **inputs, max_new_tokens=1)
        return self.tokenizer.decode(tokens[0])