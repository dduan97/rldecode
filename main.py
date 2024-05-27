import argparse
import copy
import json
from tqdm import tqdm
import random
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from models import hf_model
from metrics import gpt_judge
from tasks import tldr


def parse_args():
    parser = argparse.ArgumentParser(
        prog='RLDecode',
        description='What the program does',
        epilog='Text at the bottom of help')

    parser.add_argument('--mode', type=str,
                        choices=['eval', 'train'], help='Mode to run')
    parser.add_argument('--quantize', type=bool,
                        help='Whether to quantize the model')
    parser.add_argument('--sampling_strategy', type=str, choices=['greedy', 'temperature-policy', 'pythia-sft', 'pythia-ppo'], default='pythia-ppo',
                        help='Sampling strategy.')
    parser.add_argument('--out_dir', type=str,
                        help='Output dir', default='results')
    parser.add_argument('--model_name', type=str, help='HF model to eval.')
    parser.add_argument('--seed', type=int, default=509,
                        help='Random seed to set')

    # TRAIN FLAGS
    parser.add_argument('--train_task', type=str,
                        help='Which train task to run.')
    parser.add_argument('--train_count', type=int, default=-1,
                        help='How many train examples to run on.')
    parser.add_argument('--train_batch_size', type=int,
                        help='Batch size for training')
    parser.add_argument('--train_num_epochs', type=int,
                        help='Num epochs for training')

    # EVAL FLAGS
    parser.add_argument('--eval_task', type=str,
                        help='Which eval task to run.')
    parser.add_argument('--eval_split', type=str,
                        help='Which eval split to run.')
    parser.add_argument('--eval_count', type=int, default=-1,
                        help='How many eval examples to run on.')
    parser.add_argument('--eval_batch_size', type=int,
                        help='Batch size for evaluation')

    args = parser.parse_args()
    return args


def get_task(task: str, split: str, batch_size: int, count: int):
    if task.startswith('tldr'):
        version = task.split(':')[1]
        return tldr.Tldr(version=version, split=split, batch_size=batch_size, shuffle=False, n=count)
    else:
        raise ValueError()


def get_eval_run_name(args):
    return 'eval_{task}:{split}_{sampling_strategy}_n:{count}_bs:{bs}_model:{model}'.format(
        task=args.eval_task,
        sampling_strategy=args.sampling_strategy,
        split=args.eval_split,
        count=args.eval_count,
        bs=args.eval_batch_size,
        model=args.model_name
    )


def should_run(subdir: str):
    if os.path.exists(subdir):
        if input('Run already exists. Continue? This will overwrite data [y/n] ') == 'y':
            return True
        return False
    return True


def evaluate_model(model, dataloader, sampling_strategy, judge, output_dir):
    all_decode_queries = []
    all_decode_responses = []
    all_decode_reference_responses = []
    for _, data in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            queries = data["query_token"]
            reference_responses = data["reference_response_token"]
            # context_length = queries.shape[1]
            decode_responses = model.predict(
                queries,
                sampling_strategy=sampling_strategy
            )
            decode_queries = model.tokenizer.batch_decode(queries)
            decode_reference_responses = model.tokenizer.batch_decode(
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
    df.to_csv(f'{output_dir}/scrapes.csv')

    judge_results = judge.judge(
        df['query'], df['response'], df['reference_response'])

    with open(f'{output_dir}/rater_results.json', 'w') as f:
        json.dump(judge_results, f, ensure_ascii=False)

    return df, judge_results


def train_model(model, dataloader, *, sampling_strategy, judge, output_dir, seed, num_train_epochs):
    if sampling_strategy != 'temperature-policy':
        # Not sure what we're training (not gonna do full DPO/SFT)
        raise ValueError()

    # Use the initial model as the reference model
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    ref_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in tqdm(range(num_train_epochs)):
        for data in dataloader:
            query_responses = torch.cat(
                (data["query_chosen_token"], data["query_rejected_token"]), dim=0)
            labels = torch.cat((data["query_chosen_token_response_label"],
                               data["query_rejected_token_response_label"]), dim=0)
            with torch.no_grad():
                ref_chosen_logps, ref_rejected_logps = ref_model.forward(
                    query_responses, labels)
            chosen_logps, rejected_logps = model.forward(
                query_responses, labels)
            pi_logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            # also known as h_{\pi_\theta}^{y_w,y_l}
            logits = pi_logratios - ref_logratios
            loss = -F.logsigmoid(0.05 * logits)
            print('losses:', loss)
            loss = torch.mean(loss)
            print('loss:', loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def eval_main(args):
    run_name = get_eval_run_name(args)
    subdir = os.path.join(args.out_dir, run_name)
    if not should_run(subdir):
        return

    # Output dir
    os.makedirs(subdir, exist_ok=True)

    task = get_task(args.eval_task, split=args.eval_split,
                    batch_size=args.eval_batch_size, count=args.eval_count)
    dataloader = task.dataloader
    scrape_model = hf_model.HFModel(
        args.model_name, quantize=args.quantize)

    judge = gpt_judge.GptJudge('gpt-4o')

    scrape_df, judge_results = evaluate_model(
        scrape_model, dataloader, args.sampling_strategy, judge, subdir)
    return


def train_main(args):
    run_name = get_eval_run_name(args)
    subdir = os.path.join(args.out_dir, run_name)
    if not should_run(subdir):
        return

    # Output dir
    os.makedirs(subdir, exist_ok=True)

    train_task = get_task(args.train_task, 'train',
                          args.train_batch_size, args.train_count)
    dataloader = train_task.dataloader
    model = hf_model.HFModel(
        args.model_name, quantize=args.quantize, temperature_policy_kwargs={})

    train_model(model, dataloader, sampling_strategy=args.sampling_strategy,
                judge=gpt_judge, output_dir=subdir, seed=args.seed, num_train_epochs=args.train_num_epochs)


def main():
    args = parse_args()
    if args.mode == 'eval':
        eval_main(args)
    elif args.mode == 'train':
        train_main(args)
    else:
        raise ValueError()


main()
