import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from transformers import LogitsWarper
import wandb


class TemperaturePolicyWarper(LogitsWarper):
    def __init__(self, vocab_size: int, hidden_size: int):
        self.net = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 199)
        )  # Classification over 200 temperature choices (increments of 0.01 from 0.01 to 2)
        self.optimizer = torch.optim.Adam(self.net.parameters())

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
        del input_ids
        # scores: shape (B, vocab_size)
        # reference: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L230
        # distribution = self._distribution(scores)
        # temps = distribution.sample() # should be shape (B,)
        debug = {}
        temps = self.net(scores)
        temps = F.softmax(temps, dim=-1) # shape (B, 199)

        temps = torch.max(temps, dim=-1, keepdim=True)[0]
        temps /= 100
        # If we keep the line below, training is a bit more stable but doesn't learn
        # temps = 2 * F.sigmoid(temps)
        debug['scores'] = scores
        debug['raw_temp_net_output'] = temps

        # Weigh the policy more over time
        final_temps = (1 - policy_weight) * \
            torch.ones_like(temps) + (policy_weight * temps)
        final_temps = torch.clamp(final_temps, min=1e-3, max=2)
        debug['final_temps'] = final_temps
        scores_processed = scores / final_temps
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