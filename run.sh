# for model in vwxyzjn/EleutherAI_pythia-1b-deduped__ppo_left_padding_new_nowhiten_reward__tldr/ppo_left_padding_new_nowhiten_reward__44413__1709671965 vwxyzjn/EleutherAI_pythia-1b-deduped__ppo_left_padding_new_nowhiten_reward__tldr/ppo_left_padding_new_nowhiten_reward__55513__1709671967 vwxyzjn/EleutherAI_pythia-1b-deduped__ppo_left_padding_new_nowhiten_reward__tldr/ppo_left_padding_new_nowhiten_reward__66613__1709671965 vwxyzjn/EleutherAI_pythia-1b-deduped__ppo_left_padding_new_nowhiten_reward__tldr/ppo_left_padding_new_nowhiten_reward__77713__1709671965 ;
# model=vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr/sft__44413__1708611267
model=vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr/sft__55513__1708611267
python main.py \
    --mode train \
    --model_name $model \
    --train_task tldr:preference \
    --train_num_epochs 100 \
    --train_save_every 1000 \
    --train_batch_size 3 \
    --train_count 3000 \
    --train_grad_accumulation_steps 4 \
    --train_lr 0.0001 \
    --quantize true \
    --sampling_strategy temperature-policy \
    --shuffle true \
    --eval_count 500 \
    --eval_batch_size 1 \
    --tp_input_dim 48 \
    --tp_hidden_dim 768 \
    --tp_num_hidden_layers 4 \
    --seed 55513 \
    --tp_enable_next_token_embedding true
    # --train_total_steps 1000