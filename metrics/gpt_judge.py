import concurrent
import functools
import numpy as np
from openai import OpenAI
from typing import Any
import tqdm

from . import constants


class GptJudge():
    def __init__(self, model: str, task: str = 'tldr'):
        if task != 'tldr':
            raise ValueError('Task not supported: ' + task)
        self.task = task
        self.model = model
        self.client = OpenAI()

    def _call_once(self, prompt: str) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model,
        )
        output = chat_completion.choices[0].message.content
        return output

    def _get_judge_prompt(self, prompt: str, response_a: str, response_b: str) -> str:
        if self.task == 'tldr':
            prompt = constants.LLM_JUDGE_TLDR_PROMPT_HUANG.format(
                post=prompt, summary_a=response_a, summary_b=response_b)
            return prompt
        raise ValueError()

    def _parse_vote(self, vote: str) -> float | None:
        vote = vote.lower().strip().removeprefix('preferred: ')
        if 'a' in vote:
            return 1
        elif 'b' in vote:
            return -1
        elif 'tie' in vote:
            return 0
        return None

    def _score_to_win_count(self, score: float):
        if score < 0:
            return 0
        elif score > 0:
            return 1
        return 0.5

    def _judge_once(self, prompt: str, response_a: str, response_b: str, *, num_votes: int = 2) -> dict[str, Any]:
        scores = []
        rationales = []
        win_counts = []
        for i in range(num_votes):
            flipped = i % 2 == 1

            prompt = self._get_judge_prompt(prompt, response_b, response_a) if flipped else self._get_judge_prompt(
                prompt, response_a, response_b)

            response = self._call_once(prompt).strip()
            # Last line is the vote
            lines = response.split('\n')
            vote = lines[-1]
            vote_score = self._parse_vote(vote)
            if vote_score is None:
                print('Error in parsing GPT judge response!', vote)
                continue
            rationale = '\n'.join(lines[:-1])
            if flipped:
                vote_score = -vote_score
                rationale = '(flipped) ' + rationale
            scores.append(vote_score)
            rationales.append(rationale)
            win_counts.append(self._score_to_win_count(vote_score))
        return {
            'avg_score': np.mean(scores),
            'win_rate': np.mean(win_counts),
            'scores': scores,
            'rationales': rationales,
            'prompt': prompt,
            'response_a': response_a,
            'response_b': response_b
        }

    def judge(self, prompts: list[str], responses_a: list[str], responses_b: list[str], *, max_workers: int = 4) -> dict[str, Any]:
        responses = [None] * len(prompts)
        zipped = zip(prompts, responses_a, responses_b)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(
                self._judge_once, prompt, response_a, response_b): i for i, (prompt, response_a, response_b) in enumerate(zipped)}
            for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_idx)):
                idx = future_to_idx[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (idx, exc))
                else:
                    responses[idx] = data

        # Now generate full stats
        sxs_average = np.mean([r['avg_score'] for r in responses])
        win_rate = np.mean([r['win_rate'] for r in responses])
        return {
            'avg_score': sxs_average,
            'win_rate': win_rate,
            'details': responses
        }
