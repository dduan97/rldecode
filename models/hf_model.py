import torch
import more_itertools
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
import tqdm


from . import logits_warpers
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

    def _get_temperature_policy_warper(self):
        # return logits_warpers.TemperaturePolicyWarper(self.tokenizer.vocab_size, 2048)
        # Not sure why but the pythia model is outputting shape 50304 instead of the vocab size (50254)
        return logits_warpers.TemperaturePolicyWarper(50304, 2048)

    def predict(self, inputs: list[str], *, batch_size: int = 8, max_new_tokens=128, sampling_strategy: str = 'greedy', temperature: float = 0.0, top_p: float = 0.95) -> str:
        if sampling_strategy == 'greedy':
            generate_kwargs = {'do_sample': False}
        elif sampling_strategy == 'nucleus':
            generate_kwargs = {'temperature': temperature, 'top_p': top_p}
        elif sampling_strategy == 'temperature_policy':
            generate_kwargs = {'do_sample': True,
                               'logits_processor': LogitsProcessorList([self._get_temperature_policy_warper()])}
        else:
            raise ValueError('Invalid sampling strategy ' + sampling_strategy)
        all_decodes = []
        for batched_inputs in tqdm.tqdm(more_itertools.chunked(inputs, batch_size), desc='predict()'):
            model_inputs = self.tokenizer(
                batched_inputs, padding=True, return_tensors='pt')
            tokens = self.model.generate(
                **model_inputs, **generate_kwargs, max_new_tokens=max_new_tokens)
            # Remove the input prefix?
            tokens = tokens[:, model_inputs.input_ids.shape[1]:]
            decodes = self.tokenizer.batch_decode(tokens)
            all_decodes.extend(decodes)
        return all_decodes

    def decode(self, inputs: list[str], *, batch_size: int = 8, max_new_tokens: int = 128, return_logits: bool = False, return_int_states: bool = False, decode: bool = False):
        """Decode. Return a dict with keys:

        tokens: list of tensors
        logits: list of tensors (if `return_logits` is specified)
        int_states: list of tensors (if `return_int_states` is specified)
        decoded: list of strings (if `decode` is specified)
        """

        inputs = self.tokenizer(inputs, return_tensors='pt')
        tokens = self.model.generate(
            **inputs, max_new_tokens=1)
        return self.tokenizer.decode(tokens[0])
