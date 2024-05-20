import argparse
import json
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

    # EVAL FLAGS
    parser.add_argument('--eval_task', type=str,
                        help='Which eval task to run.')
    parser.add_argument('--eval_split', type=str,
                        help='Which eval split to run.')
    parser.add_argument('--eval_count', type=int,
                        help='How many eval examples to run on.')
    parser.add_argument('--eval_batch_size', type=int,
                        help='Batch size for evaluation')
    parser.add_argument('--eval_model', type=str, help='HF model to eval.')
    parser.add_argument('--eval_out_dir', type=str, help='Output dir to eval', default='results')

    args = parser.parse_args()
    return args


def get_task(args):
    if args.eval_task == 'tldr:axis':
        return tldr.Tldr(subset='axis', split=args.eval_split, shuffle=True, n=args.eval_count)
    else:
        raise ValueError()


def get_eval_run_name(args):
    return 'eval_{task}:{split}_n:{count}_bs:{bs}_model:{model}'.format(
        task=args.eval_task,
        split=args.eval_split,
        count=args.eval_count,
        bs=args.eval_batch_size,
        model=args.eval_model
    )


def should_run(subdir: str):
    if os.path.exists(subdir):
        if input('Run already exists. Continue? This will overwrite data [y/n]') == 'y':
            return True
        return False
    return True


def run_eval(args):
    run_name = get_eval_run_name(args)
    subdir = os.path.join(args.eval_out_dir, run_name)
    if not should_run(subdir):
        return

    task = get_task(args)
    df = task.dataset()
    df['formatted_prompt'] = df.apply(task.formatter, axis=1)
    scrape_model = hf_model.HFModel(args.eval_model, quantize=args.quantize)
    df['model_scrape'] = scrape_model.predict(
        df['formatted_prompt'], batch_size=args.eval_batch_size)

    judge = gpt_judge.GptJudge('gpt-4o')
    judge_results = judge.judge(
        df['formatted_prompt'], df['model_a_scrape'], df['summary'])

    # Save results
    os.makedirs(subdir)

    df.to_csv(f'{subdir}/scrapes.csv')
    with open(f'{subdir}/rater_results.json', 'w') as f:
        json.dump(judge_results, f, ensure_ascii=False)


def main():
    args = parse_args()
    if args.mode == 'eval':
        run_eval(args)
    else:
        raise ValueError()

main()