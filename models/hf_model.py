import torch
import more_itertools
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
import tqdm
from typing import Any


from . import logits_warpers
from . import model_base

# Mapping from path -> (tokenizer_path, model_path)
_PATH_REMAPS = {
    'CarperAI/openai_summarize_tldr_sft': ('EleutherAI/gpt-j-6b', 'CarperAI/openai_summarize_tldr_sft')
}


class HFModel(model_base.ModelBase):
    def __init__(self, model_name: str, quantize: bool = False, temperature_policy_kwargs: dict[str, Any] | None = None):
        branch = None
        path_parts = model_name.split('/')
        kwargs = {}
        if len(path_parts) == 3:
            model_name = '/'.join(path_parts[:-1])
            branch = path_parts[-1]
            print('Reading from branch', branch)
            kwargs['revision'] = branch
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **kwargs)

        # Use config from https://github.com/vwxyzjn/summarize_from_feedback_details/blob/main/summarize_from_feedback_details/sft.py#L300
        # disable `pad_token_id` and `eos_token_id` because we just want to
        self.model.generation_config.eos_token_id = None
        # generate tokens without truncation / padding
        self.model.generation_config.pad_token_id = None
        if quantize:
            self.model.to(torch.bfloat16)
        tokenizer_path, model_path = _PATH_REMAPS.get(
            model_name, (model_name, model_name))
        # Use config from https://github.com/vwxyzjn/summarize_from_feedback_details/blob/main/summarize_from_feedback_details/sft.py#L300
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            padding_side="right",
            trust_remote_code=True,
            **kwargs
        )
        # we use the padding token manually but do not resize the token embedding of the model
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.temperature_policy = None
        if temperature_policy_kwargs is not None:
            # TODO: pipe through the flags that we want
            self.temperature_policy = self._get_temperature_policy_warper()

    def _get_temperature_policy_warper(self):
        # return logits_warpers.TemperaturePolicyWarper(self.tokenizer.vocab_size, 2048)
        # Not sure why but the pythia model is outputting shape 50304 instead of the vocab size (50254)
        return logits_warpers.TemperaturePolicyWarper(50304, 2048)

    def predict(self, queries, *, max_new_tokens=128, sampling_strategy: str = 'greedy', temperature: float = 0.0, top_p: float = 0.95) -> str:
        context_length = queries.shape[1]
        attention_mask = queries != self.tokenizer.pad_token_id
        input_ids = torch.masked_fill(queries, ~attention_mask, 0)

        if sampling_strategy == 'greedy':
            generate_kwargs = {'do_sample': False,
                               'max_new_tokens': max_new_tokens}
        elif sampling_strategy == 'pythia-sft':
            generate_kwargs = {'min_new_tokens': 53, 'max_new_tokens': 53, 'do_sample': True, 'temperature': (
                0.01 + 1e-7), 'top_k': 0.0, 'top_p': 1.0}
        elif sampling_strategy == 'pythia-ppo':
            generate_kwargs = {'min_new_tokens': 53, 'max_new_tokens': 53, 'do_sample': True, 'temperature': (
                0.7 + 1e-7), 'top_k': 0.0, 'top_p': 1.0}
        elif sampling_strategy == 'nucleus':
            generate_kwargs = {'temperature': temperature, 'top_p': top_p}
        elif sampling_strategy == 'temperature-policy':
            if self.temperature_policy is None:
                raise ValueError('No temperature policy initialized')
            generate_kwargs = {'do_sample': True,
                               'logits_processor': LogitsProcessorList([self.temperature_policy])}
        else:
            raise ValueError('Invalid sampling strategy ' + sampling_strategy)
        output = self.model.generate(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     return_dict_in_generate=True,
                                     **generate_kwargs)
        generated_responses = torch.cat(
            (queries, output.sequences[:, context_length:]), dim=1)
        responses = generated_responses[:, context_length:]
        decode_responses = self.tokenizer.batch_decode(
            responses,
            skip_special_tokens=True,
        )
        return decode_responses

    def forward(self, query_responses, labels, *, policy_weight: float = 1.0):
        # DPO forward pass (on two examples)
        attention_mask = query_responses != self.tokenizer.pad_token_id
        input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        labels = labels[:, 1:].clone()
        logits = output.logits[:, :-1, :]
        # Unquantize
        logits = logits.to(torch.float)

        # Apply temperature policy
        debug = {}
        if self.temperature_policy is not None:
            # First parameter of temperature policy is not used
            logits_shape = logits.shape
            vocab_size = logits_shape[-1]
            logits = logits.reshape(-1, vocab_size)
            logits, tp_debug = self.temperature_policy(
                None, logits, return_debug=True, policy_weight=policy_weight)
            logits = logits.reshape(logits_shape)
            debug = debug | tp_debug

        loss_mask = (labels != self.tokenizer.pad_token_id)
        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        all_logps = (per_token_logps * loss_mask).sum(-1)
        chosen_logps = all_logps[:query_responses.shape[0] // 2]
        rejected_logps = all_logps[query_responses.shape[0] // 2:]
        return chosen_logps, rejected_logps, debug

    def parameters(self):
        if self.temperature_policy is not None:
            return self.temperature_policy.parameters()
        return []

    def to(self, device):
        self.model = self.model.to(device)
        if self.temperature_policy is not None:
            self.temperature_policy = self.temperature_policy.to(device)
        return self

    def get_checkpoint_info(self):
        if self.temperature_policy is not None:
            return {'type': 'temperature-policy', 'state_dict': self.temperature_policy.state_dict()}
        return {}

    def get_grad_norms(self):
        if self.temperature_policy is not None:
            return self.temperature_policy.get_grad_norms()
        return torch.Tensor()

    def get_grads(self):
        if self.temperature_policy is not None:
            return self.temperature_policy.get_grads()
        return torch.Tensor()

    def get_num_learnable_parameters(self):
        if self.temperature_policy is not None:
            return self.temperature_policy.get_num_learnable_parameters()
        return 0

    def get_parameter_norm(self):
        if self.temperature_policy is not None:
            return self.temperature_policy.get_parameter_norm()
        return 0