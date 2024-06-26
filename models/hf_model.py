import torch
import more_itertools
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, LogitsWarper
import tqdm
from typing import Any


from . import logits_warpers
from . import model_base

# Mapping from path -> (tokenizer_path, model_path)
_PATH_REMAPS = {
    'CarperAI/openai_summarize_tldr_sft': ('EleutherAI/gpt-j-6b', 'CarperAI/openai_summarize_tldr_sft')
}


class HFModel(model_base.ModelBase):
    def __init__(self, model_name: str, quantize: bool = False):
        branch = None
        # If reading a local temperature policy checkpoint, format is <HF_SPEC>:::TP:local_path
        # load the HF model
        path_parts = model_name.split('/')
        kwargs = {}
        if len(path_parts) == 3:
            model_name = '/'.join(path_parts[:-1])
            branch = path_parts[-1]
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

    def generate(self, queries, **kwargs) -> str:
        context_length = queries.shape[1]
        attention_mask = queries != self.tokenizer.pad_token_id
        input_ids = torch.masked_fill(queries, ~attention_mask, 0)

        print(kwargs)
        output = self.model.generate(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     return_dict_in_generate=True,
                                     **kwargs)
        generated_responses = torch.cat(
            (queries, output.sequences[:, context_length:]), dim=1)
        responses = generated_responses[:, context_length:]
        decode_responses = self.tokenizer.batch_decode(
            responses,
            skip_special_tokens=True,
        )
        return decode_responses

    def forward(self, query_responses, labels, *, logits_warper: LogitsWarper | None = None):
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
        if logits_warper:
            # Reshape the logits into (B, vocab_size) which is what logits warpers need
            logits_shape = logits.shape
            vocab_size = logits_shape[-1]
            logits = logits.reshape(-1, vocab_size)
            # TODO: form the proper input ids and feed in as well
            logits, tp_debug = logits_warper(
                None, logits)
            logits = logits.reshape(logits_shape)
            debug = debug | tp_debug

        loss_mask = (labels != self.tokenizer.pad_token_id)
        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        all_logps = (per_token_logps * loss_mask).sum(-1)
        chosen_logps = all_logps[:query_responses.shape[0] // 2]
        rejected_logps = all_logps[query_responses.shape[0] // 2:]
        return chosen_logps, rejected_logps, debug
    
    def get_token_embedding_layer(self):
        return self.model.gpt_neox.get_input_embeddings()

    def get_token_embedding_dim(self):
        return 128

    def to(self, device):
        self.model.to(device)
        return self