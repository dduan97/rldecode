import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from transformers import LogitsWarper
from typing import Any
import wandb


class TemperaturePolicyWarper(LogitsWarper):
    def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int, token_embedding_layer: nn.Module | Any = None, token_embedding_dim: int = None):
        # Idea: instead of taking the model logits as sparse inputs, maybe we take the topK logits, feed them through
        # some embedding layer, and then take a weighted average of them?
        self.k = input_dim
        layers = []

        token_embedding_dim = token_embedding_dim or 0

        # Input layer
        layers.extend([
            nn.Linear(input_dim + token_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        ])

        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            ])
        layers.append(
            nn.Linear(hidden_dim, 1)
        )
        self.net = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.token_embedding_layer = token_embedding_layer

        self.temps_and_inputs = {'temps': [], 'input_scores': [], 'next_token_embs': []}

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

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, *, policy_weight: float = 1., return_debug=False):
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
            token_embeddings = self.token_embedding_layer(
                topk_tokens)  # shape (B, k, H)

            weighted_reps = torch.sum(
                F.softmax(topk_scores, dim=-1).unsqueeze(-1) * token_embeddings, dim=-2)  # shape (B, H)

            # Then we concat the reps here
            model_inputs = torch.cat([model_inputs, weighted_reps], dim=-1)
            self.temps_and_inputs['next_token_embs'].append(weighted_reps.cpu().detach())

        temps = self.net(model_inputs)
        temps = F.sigmoid(temps)

        self.temps_and_inputs['temps'].append(temps.cpu().detach())
        self.temps_and_inputs['input_scores'].append(scores.cpu().detach())

        # debug['scores'] = scores
        debug['scores'] = model_inputs 
        debug['raw_temp_net_output'] = temps
        scores_processed = scores / temps
        debug['processed_scores'] = scores_processed
        if return_debug:
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
