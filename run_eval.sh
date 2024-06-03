model=vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr/sft__44413__1708611267
python main.py \
    --mode eval \
    --model_name $model \
    --eval_task tldr:sft \
    --eval_batch_size 16 \
    --eval_split validation \
    --eval_count 100 \
    --tp_state_path results/tldr:preference_temperature-policy_n:100_bs:3,None_model:vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr/sft__44413__1708611267_total_steps:1000/model_epoch29_step999.pt