import argparse
import copy
import json
import math
from tqdm import tqdm
import random
import os
import wandb

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from models import hf_model
from metrics import gpt_judge
from tasks import tldr


# if torch.cuda.is_available() else torch.device('cpu')
_DEVICE = torch.device("cuda:0")
print("Using device", _DEVICE)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="RLDecode",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument(
        "--mode", type=str, choices=["eval", "train"], help="Mode to run"
    )
    parser.add_argument("--quantize", type=bool,
                        help="Whether to quantize the model")
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        choices=["greedy", "temperature-policy", "pythia-sft", "pythia-ppo"],
        default="pythia-ppo",
        help="Sampling strategy.",
    )
    parser.add_argument("--out_dir", type=str,
                        help="Output dir", default="results")
    parser.add_argument("--model_name", type=str, help="HF model to eval.")
    parser.add_argument("--seed", type=int, default=510,
                        help="Random seed to set")
    parser.add_argument("--tp_state_path", type=str, help="path to state dict to load TP checkpoint.")

    # TRAIN FLAGS
    parser.add_argument("--train_task", type=str,
                        help="Which train task to run.")
    parser.add_argument(
        "--train_count", type=int, default=-1, help="How many train examples to run on."
    )
    parser.add_argument("--train_batch_size", type=int,
                        help="Batch size for training")
    parser.add_argument("--train_num_epochs", type=int,
                        help="Num epochs for training")
    parser.add_argument("--train_save_every", type=int, default=1000,
                        help="Num steps to save checkpoints")
    parser.add_argument("--train_total_steps", type=int, default=1000,
                        help="Total train steps to use for rampup calculations (policy weight)")
    parser.add_argument("--train_grad_accumulation_steps", type=int, default=10,
                        help="Gradient accumulation steps")

    # EVAL FLAGS
    parser.add_argument("--eval_task", type=str,
                        help="Which eval task to run.")
    parser.add_argument("--eval_split", type=str,
                        help="Which eval split to run.")
    parser.add_argument(
        "--eval_count", type=int, default=-1, help="How many eval examples to run on."
    )
    parser.add_argument("--eval_batch_size", type=int,
                        help="Batch size for evaluation")

    # Data flags
    parser.add_argument("--shuffle", type=bool,
                        help="Whether to shuffle the dataloader")

    args = parser.parse_args()
    return args


def get_task(task: str, split: str, batch_size: int, count: int, shuffle: bool):
    if task.startswith("tldr"):
        version = task.split(":")[1]
        return tldr.Tldr(
            version=version, split=split, batch_size=batch_size, shuffle=shuffle, n=count
        )
    else:
        raise ValueError()


def get_run_name(args):
    if args.mode == 'eval':
        return get_eval_name(args)
    elif args.mode == 'train':
        return get_train_name(args)

def get_eval_name(args):
    return "EVAL:{task}_{sampling_strategy}_n:{count}_bs:{eval_bs}_model:{model}_total_steps:{total_steps}".format(
        task=args.eval_task,
        sampling_strategy=args.sampling_strategy,
        count=args.eval_count,
        eval_bs=args.eval_batch_size,
        model=args.model_name,
    )


def get_train_name(args):
    return "TRAIN:{task}_{sampling_strategy}_n:{count}_bs:{train_bs},{eval_bs}_model:{model}_total_steps:{total_steps}".format(
        task=args.train_task,
        sampling_strategy=args.sampling_strategy,
        count=args.train_count,
        train_bs=args.train_batch_size,
        eval_bs=args.eval_batch_size,
        model=args.model_name,
        total_steps=args.train_total_steps
    )


def init_wandb(args, run_name):
    run = wandb.init(
        # Set the project where this run will be logged
        project="rldecode",
        # Track hyperparameters and run metadata
        name=run_name,
    )
    return run


def should_run(subdir: str):
    if os.path.exists(subdir):
        if (
            input(
                "Run already exists. Continue? This will overwrite data [y/n] ")
            == "y"
        ):
            return True
        return False
    return True


def evaluate_model(model, dataloader, sampling_strategy, judge, output_dir):
    all_decode_queries = []
    all_decode_responses = []
    all_decode_reference_responses = []
    for _, data in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            queries = data["query_token"].to(_DEVICE)
            reference_responses = data["reference_response_token"]
            # context_length = queries.shape[1]
            decode_responses = model.predict(
                queries, sampling_strategy=sampling_strategy
            )
            decode_queries = model.tokenizer.batch_decode(queries)
            decode_reference_responses = model.tokenizer.batch_decode(
                reference_responses,
                skip_special_tokens=True,
            )
            all_decode_queries.extend(decode_queries)
            all_decode_responses.extend(decode_responses)
            all_decode_reference_responses.extend(decode_reference_responses)

    df = pd.DataFrame(
        {
            "query": all_decode_queries,
            "response": all_decode_responses,
            "reference_response": all_decode_reference_responses,
        }
    )

    # Kind of janky postprocessing?
    df["query"] = df["query"].apply(lambda x: x.replace("[pad]", ""))

    # Cache the scrapes here
    df.to_csv(f"{output_dir}/scrapes.csv")

    judge_results = judge.judge(
        df["query"], df["response"], df["reference_response"])

    with open(f"{output_dir}/rater_results.json", "w") as f:
        json.dump(judge_results, f, ensure_ascii=False)

    return df, judge_results


def _policy_weight(step, max_steps):
    # val = math.tanh((6 * step / max_steps) - 3) + 1
    # return min(val, 1)
    # return min(1, step / max_steps)
    return 1


def train_model(
    model, dataloader, *, sampling_strategy, judge, output_dir, seed, num_train_epochs, save_every, wandb_run, total_steps, grad_accumulation_steps
):
    if sampling_strategy != "temperature-policy":
        # Not sure what we're training (not gonna do full DPO/SFT)
        raise ValueError()

    # Use the initial model as the reference model
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    ref_model = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters())
    global_step = 0
    for epoch in tqdm(range(num_train_epochs)):
        for data in tqdm(dataloader):
            query_responses = torch.cat(
                (data["query_chosen_token"],
                 data["query_rejected_token"]), dim=0
            )
            labels = torch.cat(
                (
                    data["query_chosen_token_response_label"],
                    data["query_rejected_token_response_label"],
                ),
                dim=0,
            )
            policy_weight = _policy_weight(global_step, total_steps)
            query_responses = query_responses.to(_DEVICE)
            labels = labels.to(_DEVICE)
            with torch.no_grad():
                ref_chosen_logps, ref_rejected_logps, _ = ref_model.forward(
                    query_responses, labels
                )
            chosen_logps, rejected_logps, debug = model.forward(
                query_responses, labels, policy_weight=policy_weight)
            pi_logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            # also known as h_{\pi_\theta}^{y_w,y_l}
            logits = pi_logratios - ref_logratios
            loss = -F.logsigmoid(0.05 * logits)
            loss = torch.mean(loss)
            loss.backward()
            log_data = {
                'loss': loss,
                'final_temps': wandb.Histogram(debug['final_temps'].cpu().detach()),
                'raw_temp_net_output': wandb.Histogram(debug['raw_temp_net_output'].cpu().detach()),
                'scores': wandb.Histogram(debug['scores'].cpu().detach()),
                'policy_weight': policy_weight,
                'num_learnable_params': model.get_num_learnable_parameters(),
                'parameter_norm': model.get_parameter_norm(),
                'processed_scores': wandb.Histogram(debug['processed_scores'].cpu().detach()),
                'ref_chosen_logps': wandb.Histogram(ref_chosen_logps.cpu().detach()),
                'ref_rejected_logps': wandb.Histogram(ref_rejected_logps.cpu().detach()),
                'chosen_logps': wandb.Histogram(chosen_logps.cpu().detach()),
                'rejected_logps': wandb.Histogram(rejected_logps.cpu().detach()),
            }
            # grads = model.get_grads().cpu().detach()
            # if grads is not None and grads.numel():
            #     log_data['grads'] = wandb.Histogram(grads)

            grad_norms = model.get_grad_norms().cpu().detach()
            if grad_norms is not None and grad_norms.numel():
                log_data['grad_norms'] = wandb.Histogram(grad_norms)
            wandb_run.log(log_data, step=global_step)

            if (global_step + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                # for param in model.parameters():
                #     param.grad /= grad_accumulation_steps

            if (global_step + 1) % save_every == 0:
                # Save a checkpoint for the temperature policy every epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.get_checkpoint_info(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f'{output_dir}/model_epoch{epoch}_step{global_step}.pt')
            global_step += 1

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.get_checkpoint_info(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'{output_dir}/model_final.pt')


def eval_main(args):
    run_name = get_run_name(args)
    wandb_run = init_wandb(args, run_name)
    subdir = os.path.join(args.out_dir, run_name)
    if not should_run(subdir):
        return

    # Output dir
    os.makedirs(subdir, exist_ok=True)

    task = get_task(
        args.eval_task,
        split=args.eval_split,
        batch_size=args.eval_batch_size,
        count=args.eval_count,
        shuffle=args.shuffle
    )
    dataloader = task.dataloader
    tp_kwargs = {}
    if args.tp_state_path:
        state_dict = torch.load(args.tp_state_path)
        tp_kwargs['state_dict'] = state_dict['model_state_dict']['state_dict']
    scrape_model = hf_model.HFModel(args.model_name, quantize=args.quantize, temperature_policy_kwargs=tp_kwargs)

    scrape_model.to(_DEVICE)

    judge = gpt_judge.GptJudge("gpt-4o")

    scrape_df, judge_results = evaluate_model(
        scrape_model, dataloader, args.sampling_strategy, judge, subdir
    )
    return


def train_main(args):
    run_name = get_run_name(args)
    subdir = os.path.join(args.out_dir, run_name)
    if not should_run(subdir):
        return

    wandb_run = init_wandb(args, run_name)
    # Output dir
    os.makedirs(subdir, exist_ok=True)

    train_task = get_task(
        args.train_task, "train", args.train_batch_size, args.train_count, shuffle=args.shuffle
    )
    dataloader = train_task.dataloader
    tp_kwargs = {}
    if args.tp_state_path:
        state_dict = torch.load(args.tp_state_path)
        tp_kwargs['state_dict'] = state_dict['model_state_dict']['state_dict']
    model = hf_model.HFModel(
        args.model_name, quantize=args.quantize, temperature_policy_kwargs=tp_kwargs
    )
    model.to(_DEVICE)

    train_model(
        model,
        dataloader,
        sampling_strategy=args.sampling_strategy,
        judge=gpt_judge,
        output_dir=subdir,
        seed=args.seed,
        num_train_epochs=args.train_num_epochs,
        save_every=args.train_save_every,
        wandb_run=wandb_run,
        total_steps=args.train_total_steps,
        grad_accumulation_steps=args.train_grad_accumulation_steps
    )


def main():
    args = parse_args()
    if args.mode == "eval":
        eval_main(args)
    elif args.mode == "train":
        train_main(args)
    else:
        raise ValueError()


main()
