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
from transformers import LogitsProcessorList

from models import hf_model
from models import logits_warpers
from metrics import gpt_judge
from tasks import tldr


# if torch.cuda.is_available() else torch.device('cpu')
# _DEVICE = torch.device("cuda:0")
_DEVICE = torch.device("cpu")
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
        # choices=["greedy", "temperature-policy", "pythia-sft", "pythia-ppo"],
        default="pythia-ppo",
        help="Sampling strategy.",
    )
    parser.add_argument("--out_dir", type=str,
                        help="Output dir", default="results")
    parser.add_argument("--model_name", type=str, help="HF model to eval.")
    parser.add_argument("--seed", type=int, default=510,
                        help="Random seed to set")
    parser.add_argument("--tp_state_path", type=str,
                        help="path to state dict to load TP checkpoint.")

    # TRAIN FLAGS
    parser.add_argument("--train_task", type=str,
                        help="Which train task to run.")
    parser.add_argument(
        "--train_count", type=int, default=-1, help="How many train examples to run on."
    )
    parser.add_argument(
        "--train_lr", type=float, default=1e-3, help="Learning rate"
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

    parser.add_argument("--tp_top_k", type=int, default=64,
                        help="top k to use for temperature policy")
    parser.add_argument("--tp_hidden_dims", type=int, nargs='+', default=[512, 512],
                        help="temperature policy hidden dim")
    parser.add_argument("--tp_return_debug", type=bool, default=True,
                        help="Whether to return debug info from logits warpers")
    parser.add_argument("--tp_store_logits", type=bool, default=True,
                        help="Whether to accumulate and store logits (memory intensive)")

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
    parser.add_argument(
        "--eval_judge", type=str, help="Which judge to use"
    )

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


def get_logits_warper(args, hf_model):
    if args.sampling_strategy == 'temperature-policy':
        logits_warper = logits_warpers.TemperaturePolicyWarper(args.tp_top_k, args.tp_hidden_dims, hf_model.get_token_embedding_layer(), hf_model.get_token_embedding_dim(), args.tp_return_debug, args.tp_store_logits)
        if args.tp_state_path:
            state_dict = torch.load(args.tp_state_path)
            logits_warper_state = state_dict['logits_warper_state']
            assert logits_warper_state['type'] == 'temperature-policy'
            logits_warper.load_state_dict(logits_warper_state['state_dict'])
        
        if args.mode == 'eval':
            logits_warper.set_debug(False)
        return logits_warper
            
    elif args.sampling_strategy == 'edt':
        return logits_warpers.EdtWarper()
    else:
        return None


def get_run_name(args):
    if args.mode == 'eval':
        return get_eval_name(args)
    elif args.mode == 'train':
        return get_train_name(args)


def get_eval_name(args):
    return "EVAL:{task}_{sampling_strategy}_judge:{judge}_n:{count}_bs:{eval_bs}_ckpt:{ckpt}_seed:{seed}_model:{model}".format(
        ckpt=args.tp_state_path.split('/')[-1] if args.tp_state_path else 'pt',
        task=args.eval_task,
        sampling_strategy=args.sampling_strategy,
        count=args.eval_count,
        eval_bs=args.eval_batch_size,
        model=args.model_name,
        seed=args.seed,
        judge=args.eval_judge,
    )


def get_train_name(args):
    return "TRAIN:{task}_{sampling_strategy}_n:{count}_valn:{val_count}_bs:{train_bs}_tpspecs:{topk},{hidden_dims}_seed:{seed}_model:{model}".format(
        task=args.train_task,
        sampling_strategy=args.sampling_strategy,
        count=args.train_count,
        train_bs=args.train_batch_size,
        eval_bs=args.eval_batch_size,
        model=args.model_name,
        topk=args.tp_top_k,
        hidden_dims=','.join([str(d) for d in args.tp_hidden_dims]),
        val_count=args.eval_count,
        seed=args.seed,
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


def run_validation_metrics(model, logits_warper, dataloader, wandb_run, global_step: int):
    print('VALIDATION')
    all_validation_accs = []
    all_validation_losses = []
    all_validation_final_temps = []
    all_validation_processed_scores = []
    for _, data in tqdm(enumerate(dataloader)):
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
        query_responses = query_responses.to(_DEVICE)
        labels = labels.to(_DEVICE)
        logits_warper.eval()
        with torch.no_grad():
            ref_chosen_logps, ref_rejected_logps, _ = model.forward(
                query_responses, labels
            )
            chosen_logps, rejected_logps, debug = model.forward(
                query_responses, labels, logits_warper=logits_warper)
            pi_logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            # Compute implcit DPO reward/accuracy
            reward_preferred = 0.05 * (chosen_logps - ref_chosen_logps)
            reward_rejected = 0.05 * (rejected_logps - ref_rejected_logps)

            accuracy = (reward_preferred > reward_rejected).float().mean()
            # also known as h_{\pi_\theta}^{y_w,y_l}
            logits = pi_logratios - ref_logratios
            loss = -F.logsigmoid(0.05 * logits)
            loss = torch.mean(loss)

            all_validation_losses.append(loss.cpu().detach())
            all_validation_final_temps.append(debug['temps'].cpu().detach())
            all_validation_processed_scores.append(
                debug['processed_scores'].cpu().detach())
            all_validation_accs.append(accuracy.cpu().detach())
        logits_warper.train()
    log_data = {
        'validation/loss': np.mean(all_validation_losses),
        'validation/temps': wandb.Histogram(torch.stack(all_validation_final_temps)),
        'validation/processed_scores': wandb.Histogram(torch.stack(all_validation_processed_scores)),
        'validation/accuracy': np.mean(all_validation_accs),
    }
    wandb_run.log(log_data, step=global_step)


def evaluate_model(model, logits_warper, dataloader, sampling_strategy, judge, output_dir):
    all_decode_queries = []
    all_decode_responses = []
    all_decode_reference_responses = []
    for _, data in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            queries = data["query_token"].to(_DEVICE)
            reference_responses = data["reference_response_token"]
            # context_length = queries.shape[1]
            decode_responses = model.generate(
                queries, do_sample=True, max_new_tokens=53, logits_processor=LogitsProcessorList([logits_warper])
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
    df["query"] = df["query"].apply(lambda x: x.replace("[PAD]", ""))

    # Cache the scrapes here
    df.to_csv(f"{output_dir}/scrapes.csv")

    # Write out the temperatures/scores
    torch.save(logits_warper.temps_and_inputs,
               f'{output_dir}/temperature_policy_logs.pt')

    judge_results = judge.judge(
        df["query"], df["response"], df["reference_response"])

    with open(f"{output_dir}/rater_results.json", "w") as f:
        json.dump(judge_results, f)

    return df, judge_results


def _policy_weight(step, max_steps):
    # val = math.tanh((6 * step / max_steps) - 3) + 1
    # return min(val, 1)
    # return min(1, step / max_steps)
    return 1


def train_model(
    model, dataloader, *, logits_warper, validation_dataloader, sampling_strategy, judge, output_dir, seed, num_train_epochs, save_every, wandb_run, total_steps, grad_accumulation_steps, lr
):
    if sampling_strategy != "temperature-policy":
        # Not sure what we're training (not gonna do full DPO/SFT)
        raise ValueError()

    # Use the initial model as the reference model

    # TODO: redesign HFModel to take in a pretrained LLM and an optional TP
    model.model = model.model.eval()

    optimizer = torch.optim.Adam(logits_warper.parameters(), lr=lr)
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
                ref_chosen_logps, ref_rejected_logps, _ = model.forward(
                    query_responses, labels
                )
            chosen_logps, rejected_logps, debug = model.forward(
                query_responses, labels, logits_warper=logits_warper)
            pi_logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps

            # Compute implcit DPO reward/accuracy
            reward_preferred = 0.05 * (chosen_logps - ref_chosen_logps)
            reward_rejected = 0.05 * (rejected_logps - ref_rejected_logps)

            accuracy = (reward_preferred > reward_rejected).float().mean()
            # also known as h_{\pi_\theta}^{y_w,y_l}
            logits = pi_logratios - ref_logratios
            loss = -F.logsigmoid(0.05 * logits)
            loss = torch.mean(loss)
            loss.backward()
            topk_scores = debug['scores'].cpu().detach()
            # topk_scores = torch.softmax(topk_scores, dim=-1)
            min_topk_scores = topk_scores[:, -1]
            max_topk_scores = topk_scores[:, 0]
            log_data = {
                'loss': loss,
                'raw_temp_net_output': wandb.Histogram(debug['raw_temp_net_output'].cpu().detach()),
                'scores': wandb.Histogram(debug['scores'].cpu().detach()),
                'policy_weight': policy_weight,
                'num_learnable_params': logits_warper.get_num_learnable_parameters(),
                'parameter_norm': logits_warper.get_parameter_norm(),
                'processed_scores': wandb.Histogram(debug['processed_scores'].cpu().detach()),
                'ref_chosen_logps': wandb.Histogram(ref_chosen_logps.cpu().detach()),
                'ref_rejected_logps': wandb.Histogram(ref_rejected_logps.cpu().detach()),
                'chosen_logps': wandb.Histogram(chosen_logps.cpu().detach()),
                'rejected_logps': wandb.Histogram(rejected_logps.cpu().detach()),
                'train_accuracy': accuracy,
                'min_topk_score': wandb.Histogram(min_topk_scores),
                'max_topk_scores': wandb.Histogram(max_topk_scores),
            }
            # grads = model.get_grads().cpu().detach()
            # if grads is not None and grads.numel():
            #     log_data['grads'] = wandb.Histogram(grads)

            grad_norms = logits_warper.get_grad_norms().cpu().detach()
            if grad_norms is not None and grad_norms.numel():
                log_data['grad_norms'] = wandb.Histogram(grad_norms)
            wandb_run.log(log_data, step=global_step)

            if (global_step + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                # for param in model.parameters():
                #     param.grad /= grad_accumulation_steps

            if (global_step + 1) % save_every == 0:
                # Save a checkpoint for the temperature policy
                torch.save({
                    'epoch': epoch,
                    'logits_warper_state': logits_warper.get_checkpoint_info(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f'{output_dir}/model_epoch{epoch}_step{global_step}.pt')

                # Also run validation
                # if validation_dataloader:
                #     run_validation_metrics(
                #         model, validation_dataloader, wandb_run=wandb_run, global_step=global_step)
            global_step += 1

    torch.save({
        'logits_warper_state': logits_warper.get_checkpoint_info(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'{output_dir}/model_final.pt')


def eval_main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

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

    scrape_model = hf_model.HFModel(
        args.model_name, quantize=args.quantize)
    scrape_model.to(_DEVICE)

    logits_warper = get_logits_warper(args, scrape_model)

    judge = gpt_judge.GptJudge(args.eval_judge)

    scrape_df, judge_results = evaluate_model(
        scrape_model, logits_warper, dataloader, args.sampling_strategy, judge, subdir
    )
    return


def train_main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

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
    validation_task = get_task(
        args.train_task, "validation", args.eval_batch_size, args.eval_count, shuffle=False
    )
    # validation_dataloader = validation_task.dataloader
    validation_dataloader = None

    model = hf_model.HFModel(
        args.model_name, quantize=args.quantize
    )
    model.to(_DEVICE)

    logits_warper = get_logits_warper(args, model)

    train_model(
        model,
        dataloader,
        logits_warper=logits_warper,
        validation_dataloader=validation_dataloader,
        sampling_strategy=args.sampling_strategy,
        judge=gpt_judge,
        output_dir=subdir,
        seed=args.seed,
        num_train_epochs=args.train_num_epochs,
        save_every=args.train_save_every,
        wandb_run=wandb_run,
        total_steps=args.train_total_steps,
        grad_accumulation_steps=args.train_grad_accumulation_steps,
        lr=args.train_lr
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
