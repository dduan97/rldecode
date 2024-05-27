import torch
from torch import nn
import torch.distributions as D
from transformers import LogitsWarper


class TemperaturePolicyWarper(LogitsWarper):
    def __init__(self, vocab_size: int, hidden_size: int):
        self.net = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )
        self.optimizer = torch.optim.Adam(self.net.parameters())

    def _distribution(self, scores: torch.FloatTensor) -> torch.distributions.Distribution:
        preds = self.net(scores) # shape (B, 2)
        
        mean = preds[:, 0]  # shape (B,)
        log_stdev = preds[:, 1]  # shape (B,)
        stdev = torch.exp(log_stdev)

        dist = D.Normal(mean, stdev)
        # return D.Independent(dist, 2)
        return dist
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del input_ids
        # scores: shape (B, vocab_size)
        # reference: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L230
        distribution = self._distribution(scores)
        temps = distribution.sample() # should be shape (B,)
        # Clip the temps
        temps = torch.clip(temps, min=0.001, max=3.0).unsqueeze(-1)
        scores_processed = scores / temps
        return scores_processed

    def parameters(self):
        return self.net.parameters()