import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from transformers import LogitsWarper
from typing import Any
import wandb


class TemperaturePolicyWarper(LogitsWarper):
    def __init__(self, top_k: int, hidden_dims: list[int], token_embedding_layer: nn.Module | Any = None, token_embedding_dim: int = None, return_debug: bool = False, store_logits: bool = False):
        """
        Args:
            top_k: top k tokens' logits will be passed in
            hidden_dims: list of hidden dims to use
            token_embedding_layer: an nn.module to use for embedding tokens. If this is set, token_embedding_dim should also be set
            token_embedding_dim: output dim for token_embedding_layer
            return_debug: if true, then at every call to __call__, this object will return a dict with various debug info
            store_logits: if true, this object will accumulate logits and model scores for every call to __call__. Very memory intensive.
        """
        if token_embedding_layer and not token_embedding_dim:
            raise ValueError(
                'Need token embedding dim if token_embedding_layer is set')
        # Idea: instead of taking the model logits as sparse inputs, maybe we take the topK logits, feed them through
        # some embedding layer, and then take a weighted average of them?
        self.k = top_k
        self.net = self._initialize_net(
            top_k, hidden_dims, token_embedding_dim)

        # Token embedding layer
        self.token_embedding_layer = token_embedding_layer

        self.return_debug = return_debug
        self.store_logits = store_logits
        self.temps_and_inputs = {'temps': [],
                                 'input_scores': [], 'next_token_embs': []}

    def _initialize_net(self, top_k: int, hidden_dims: list[int], token_embedding_dim: int) -> nn.Module:
        layers = []

        token_embedding_dim = token_embedding_dim or 0
        input_dim = top_k + token_embedding_dim
        print('Initializing temperature policy with input dim', input_dim)
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            dim1, dim2 = dims[i], dims[i+1]
            layers.extend([
                nn.Linear(dim1, dim2),
                nn.ReLU(),
                nn.BatchNorm1d(dim2),
            ])

        # Output dim
        layers.append(
            nn.Linear(dims[-1], 1)
        )
        return nn.Sequential(*layers)

    def _distribution(self, scores: torch.FloatTensor) -> torch.distributions.Distribution:
        preds = self.net(scores)  # shape (B, 2)

        mean = preds[:, 0]  # shape (B,)
        mean = 2 * F.sigmoid(mean)  # Rescale to (0, 2)
        log_stdev = preds[:, 1]  # shape (B,)
        log_stdev = torch.clamp(log_stdev, min=-100.0, max=1.0)
        stdev = torch.exp(log_stdev)

        dist = D.Normal(mean, stdev)
        # return D.Independent(dist, 2)
        return dist

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del input_ids
        # scores: shape (B, vocab_size)
        # reference: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L230
        # distribution = self._distribution(scores)
        # temps = distribution.sample() # should be shape (B,)

        debug = {}
        # Shape (B, k)
        topk_scores, topk_tokens = torch.topk(scores, self.k, dim=-1)

        sorted_topk_scores = torch.sort(
            topk_scores, descending=True, dim=-1)[0]
        model_inputs = F.softmax(sorted_topk_scores, dim=-1)  # shape (B, k)

        if self.token_embedding_layer:
            # topk_scores is shape (B, k)
            # topk_tokens is shape (B, k)
            with torch.no_grad():
                token_embeddings = self.token_embedding_layer(
                    topk_tokens)  # shape (B, k, H)

            weighted_reps = torch.sum(
                F.softmax(topk_scores, dim=-1).unsqueeze(-1) * token_embeddings, dim=-2)  # shape (B, H)

            # Then we concat the reps here
            model_inputs = torch.cat([model_inputs, weighted_reps], dim=-1)
            if self.store_logits:
                self.temps_and_inputs['next_token_embs'].append(
                    weighted_reps.cpu().detach())

        temps = self.net(model_inputs)
        temps = F.sigmoid(temps)

        if self.store_logits:
            self.temps_and_inputs['temps'].append(temps.cpu().detach())
            self.temps_and_inputs['input_scores'].append(scores.cpu().detach())

        # debug['scores'] = scores
        debug['scores'] = model_inputs
        debug['raw_temp_net_output'] = temps
        scores_processed = scores / temps
        debug['processed_scores'] = scores_processed
        if self.return_debug:
            return scores_processed, debug
        return scores_processed

    def parameters(self):
        return self.net.parameters()

    def to(self, device):
        self.net = self.net.to(device)
        return self

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, state_dict: dict):
        print('Loading state dict')
        self.net.load_state_dict(state_dict)
        return self

    def get_grad_norms(self):
        # distribution of norm of grads over each parameter
        parameters = [p for p in self.net.parameters(
        ) if p.grad is not None and p.requires_grad]
        if len(parameters) == 0:
            norms = torch.Tensor()
        else:
            device = parameters[0].grad.device
            norms = torch.stack(
                [torch.norm(p.grad.detach()).to(device) for p in parameters])
        return norms

    def get_num_learnable_parameters(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def get_parameter_norm(self):
        parameters = [p.reshape(-1) for p in self.net.parameters(
        ) if p.grad is not None and p.requires_grad]
        if not parameters:
            return torch.Tensor()
        return torch.linalg.vector_norm(torch.cat(parameters, dim=0).reshape(-1), ord=2)

    def eval(self):
        self.net.eval()

    def set_debug(self, debug: bool):
        self.return_debug = debug

    def get_checkpoint_info(self):
        return {
            'type': 'temperature-policy',
            'state_dict': self.net.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)
        return self


class EdtWarper(LogitsWarper):
    # https://arxiv.org/pdf/2403.14541
    def __init__(self, t0: float = 1.0, theta: float = 2.0, N: float = 0.8):
        self.t0 = t0
        self.theta = theta
        self.N = 0.8

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del input_ids
        # shape (B, vocab_size)
        # Compute entropies
        probs = torch.softmax(scores, dim=1)
        entropies = -torch.sum(probs * torch.log2(probs),
                               dim=1, keepdim=True)  # shape (B, 1)
        exponents = self.theta / entropies
        temperatures = torch.pow(self.N, exponents) * self.t0  # shape (B, 1)
        # Clip it just in case
        temperatures = torch.clamp(temperatures, min=1e-7, max=1.0)
        # print(temperatures)
        scores /= temperatures
        return scores
