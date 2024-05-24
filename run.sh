# for model in vwxyzjn/EleutherAI_pythia-1b-deduped__ppo_left_padding_new_nowhiten_reward__tldr/ppo_left_padding_new_nowhiten_reward__44413__1709671965 ; 
for model in vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr/sft__44413__1708611267 ; 
do
    python main.py \
    --mode eval \
    --eval_task tldr:sft \
    --eval_split validation \
    --eval_model $model \
    --eval_batch_size=32
done