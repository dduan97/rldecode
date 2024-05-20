import torch
import more_itertools
from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding
import tqdm

from . import model_base

# Mapping from path -> (tokenizer_path, model_path)
_PATH_REMAPS = {
    'CarperAI/openai_summarize_tldr_sft': ('EleutherAI/gpt-j-6b', 'CarperAI/openai_summarize_tldr_sft')
}


class HFModel(model_base.ModelBase):
    def __init__(self, model_name: str, quantize: bool = False):
        tokenizer_path, model_path = _PATH_REMAPS.get(
            model_name, (model_name, model_name))
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path)
        if quantize:
            self.model.to(torch.bfloat16)

    def predict(self, inputs: list[str], *, batch_size: int = 8, max_new_tokens=128) -> str:
        all_decodes = []
        for batched_inputs in tqdm.tqdm(more_itertools.chunked(inputs, batch_size), desc='predict()'):
            model_inputs = self.tokenizer(batched_inputs, padding=True, return_tensors='pt')
            tokens = self.model.generate(
                **model_inputs, max_new_tokens=max_new_tokens)
            # Remove the input prefix?
            tokens = tokens[:, model_inputs.input_ids.shape[1]:]
            decodes = self.tokenizer.batch_decode(tokens)
            all_decodes.extend(decodes)
        return all_decodes

    def decode_step(self, inputs: BatchEncoding) -> str:
        inputs = self.tokenizer(inputs, return_tensors='pt')
        tokens = self.model.generate(
            **inputs, max_new_tokens=1)
        return self.tokenizer.decode(tokens[0])
