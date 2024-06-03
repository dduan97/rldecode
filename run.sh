# for model in vwxyzjn/EleutherAI_pythia-1b-deduped__ppo_left_padding_new_nowhiten_reward__tldr/ppo_left_padding_new_nowhiten_reward__44413__1709671965 vwxyzjn/EleutherAI_pythia-1b-deduped__ppo_left_padding_new_nowhiten_reward__tldr/ppo_left_padding_new_nowhiten_reward__55513__1709671967 vwxyzjn/EleutherAI_pythia-1b-deduped__ppo_left_padding_new_nowhiten_reward__tldr/ppo_left_padding_new_nowhiten_reward__66613__1709671965 vwxyzjn/EleutherAI_pythia-1b-deduped__ppo_left_padding_new_nowhiten_reward__tldr/ppo_left_padding_new_nowhiten_reward__77713__1709671965 ;
model=vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr/sft__44413__1708611267
python main.py \
    --mode train \
    --model_name $model \
    --train_task tldr:preference \
    --train_num_epochs 100 \
    --train_count 500 \
    --train_save_every 250 \
    --train_batch_size 4 \
    --quantize true \
    --sampling_strategy temperature-policy \
    --train_grad_accumulation_steps 8 \
    --shuffle true \
    # --train_total_steps 1000