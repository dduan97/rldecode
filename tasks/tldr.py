import pandas as pd
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset

_PRETRAINED_TEMPLATE = """POST:
{post}

TLDR:
"""

_INSTRUCTION_TEMPLATE = """ SUBREDDIT: r/{subreddit}
TITLE: {title}
POST: {post}
TL;DR:"""


class Tldr:
    def __init__(self, version: str = 'sft', split: str = 'validation', batch_size: int = 8, *, shuffle=False, seed: int = 509, n: int = -1):
        # Subset is either 'sft' or 'preference'
        # Use the processed version from https://arxiv.org/pdf/2403.17031, since we're also going to use their trained checkpoints.
        if version == 'sft':
            dataset = load_dataset(
                'vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144')[split]
        elif version == 'preference':
            dataset = load_dataset(
                'vwxyzjn/summarize_from_feedback_oai_preprocessing_1706381144')[split]
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        if n > 0:
            dataset = dataset[:n]

        # dataset = Dataset.from_dict(dataset)
        dataset = dataset.with_format(
            "torch", columns=["query_token", "reference_response_token"])
        self.dataloader = DataLoader(
            dataset, batch_size=batch_size)