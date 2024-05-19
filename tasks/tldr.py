import pandas as pd
from datasets import load_dataset

_PRETRAINED_TEMPLATE = """POST:
{post}

TLDR:
"""

_INSTRUCTION_TEMPLATE = """You will be given a forum post, and your job is to generate a concise summary of it. Include all of the important details without including any irrelevant information.

FORUM POST:
{post}

YOUR SUMMARY:
"""


class Tldr:
    def __init__(self, subset: str = 'comparisons', split: str = 'validation', *, shuffle=False, seed: int = 509, n: int = -1, mode='pretrained'):
        # Subset is either 'axis' or 'comparisons'
        dataset = load_dataset('openai/summarize_from_feedback', subset)[split]
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        if n > 0:
            dataset = dataset[:n]
        self.ds = pd.DataFrame(dataset)
        self.mode = mode

    def dataset(self) -> pd.DataFrame:
        return self.ds

    def formatter(self, example: pd.DataFrame) -> str:
        # Example should be a row from dataset()
        post = example['info']['post']
        if self.mode == 'pretrained':
            return _PRETRAINED_TEMPLATE.format(post=post)
        elif self.mode == 'instruct':
            return _INSTRUCTION_TEMPLATE.format(post=post)
