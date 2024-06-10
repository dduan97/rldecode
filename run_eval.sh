    model=vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr/sft__44413__1708611267
    # model=vwxyzjn/EleutherAI_pythia-1b-deduped__ppo_left_padding_new_nowhiten_reward__tldr/ppo_left_padding_new_nowhiten_reward__44413__1709671965
    python main.py \
        --mode eval \
        --model_name $model \
        --eval_task tldr:sft \
        --eval_batch_size 8 \
        --eval_count 500 \
        --sampling_strategy edt \
        --eval_split validation \
        --eval_judge gpt-4o
        # --tp_state_path results/TRAIN:tldr:preference_temperature-policy_n:3000_valn:500_bs:3_model:vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr/sft__44413__1708611267_tpspecs:48,768,4,True_seed:44413/model_epoch30_step30999.pt \