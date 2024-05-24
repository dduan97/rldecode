import argparse
import json
import torch
from tqdm import tqdm
import pandas as pd
import os

from models import hf_model
from metrics import gpt_judge
from tasks import tldr


def parse_args():
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')

    parser.add_argument('--mode', type=str,
                        choices=['eval'], help='Mode to run')
    parser.add_argument('--quantize', type=bool,
                        help='Whether to quantize the model')
    parser.add_argument('--sampling_strategy', type=str, choices=['greedy', 'temperature_policy', 'pythia-default'], default='pythia-default',
                        help='Sampling strategy.')

    # EVAL FLAGS
    parser.add_argument('--eval_task', type=str,
                        help='Which eval task to run.')
    parser.add_argument('--eval_split', type=str,
                        help='Which eval split to run.')
    parser.add_argument('--eval_count', type=int, default=-1,
                        help='How many eval examples to run on.')
    parser.add_argument('--eval_batch_size', type=int,
                        help='Batch size for evaluation')
    parser.add_argument('--eval_model', type=str, help='HF model to eval.')
    parser.add_argument('--eval_out_dir', type=str,
                        help='Output dir to eval', default='results')

    args = parser.parse_args()
    return args


def get_task(args):
    if args.eval_task == 'tldr:sft':
        return tldr.Tldr(version='sft', split=args.eval_split, batch_size=args.eval_batch_size, shuffle=False, n=args.eval_count)
    else:
        raise ValueError()


def get_eval_run_name(args):
    return 'eval_{task}:{split}_{sampling_strategy}_n:{count}_bs:{bs}_model:{model}'.format(
        task=args.eval_task,
        sampling_strategy=args.sampling_strategy,
        split=args.eval_split,
        count=args.eval_count,
        bs=args.eval_batch_size,
        model=args.eval_model
    )


def should_run(subdir: str):
    if os.path.exists(subdir):
        if input('Run already exists. Continue? This will overwrite data [y/n] ') == 'y':
            return True
        return False
    return True


def run_eval(args):
    run_name = get_eval_run_name(args)
    subdir = os.path.join(args.eval_out_dir, run_name)
    if not should_run(subdir):
        return

    # Output dir
    os.makedirs(subdir, exist_ok=True)

    task = get_task(args)
    dataloader = task.dataloader
    scrape_model = hf_model.HFModel(args.eval_model, quantize=args.quantize)

    all_decode_queries = []
    all_decode_responses = []
    all_decode_reference_responses = []
    for _, data in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            queries = data["query_token"]
            reference_responses = data["reference_response_token"]
            # context_length = queries.shape[1]
            decode_responses = scrape_model.predict(
                queries,
                sampling_strategy=args.sampling_strategy
            )
            decode_queries = scrape_model.tokenizer.batch_decode(queries)
            decode_reference_responses = scrape_model.tokenizer.batch_decode(
                reference_responses,
                skip_special_tokens=True,
            )
            all_decode_queries.extend(decode_queries)
            all_decode_responses.extend(decode_responses)
            all_decode_reference_responses.extend(decode_reference_responses)

    df = pd.DataFrame({'query': all_decode_queries, 'response': all_decode_responses,
                      'reference_response': all_decode_reference_responses})

    # Kind of janky postprocessing?
    df['query'] = df['query'].apply(lambda x: x.replace('[pad]', ''))

    # Cache the scrapes here
    df.to_csv(f'{subdir}/scrapes.csv')

    judge = gpt_judge.GptJudge('gpt-4o')
    judge_results = judge.judge(
        df['query'], df['response'], df['reference_response'])

    with open(f'{subdir}/rater_results.json', 'w') as f:
        json.dump(judge_results, f, ensure_ascii=False)


def main():
    args = parse_args()
    if args.mode == 'eval':
        run_eval(args)
    else:
        raise ValueError()


main()
