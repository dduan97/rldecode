    # # model=vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr/sft__44413__1708611267
    # model=vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr/sft__55513__1708611267
    # # model=vwxyzjn/EleutherAI_pythia-1b-deduped__ppo_left_padding_new_nowhiten_reward__tldr/ppo_left_padding_new_nowhiten_reward__55513__1709671967
    # python main.py \
    #     --mode eval \
    #     --model_name $model \
    #     --eval_task tldr:sft \
    #     --eval_batch_size 8 \
    #     --eval_count 500 \
    #     --sampling_strategy temperature-policy \
    #     --eval_split validation \
    #     --eval_judge gpt-4o \
    #     --tp_state_path results/TRAIN:tldr:preference_temperature-policy_n:3000_valn:500_bs:3_model:vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr/sft__55513__1708611267_tpspecs:48,768,4,True_seed:55513/model_epoch${epoch}_step${epoch}999.pt 

    # model=vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr/sft__44413__1708611267
    # model=vwxyzjn/EleutherAI_pythia-1b-deduped__ppo_left_padding_new_nowhiten_reward__tldr/ppo_left_padding_new_nowhiten_reward__44413__1709671965
    model=EleutherAI/pythia-14m
    python main.py \
        --mode eval \
        --model_name $model \
        --eval_task tldr:sft \
        --eval_batch_size 8 \
        --eval_count 8 \
        --sampling_strategy temperature-policy \
        --eval_split validation \
        --eval_judge gpt-4o \
        --tp_top_k 48 \
        --tp_hidden_dims 1024 512 128 \
        --tp_state_path results/TRAIN:tldr:preference_temperature-policy_n:3000_valn:500_bs:3_tpspecs:48,1024,512,128_seed:55513_model:EleutherAI/pythia-14m/model_epoch0_step0.pt
        #  --tp_state_path results/TRAIN:tldr:preference_temperature-policy_n:3000_valn:500_bs:3_model:vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr/sft__55513__1708611267_tpspecs:48,768,4,True_seed:55513/model_epoch25_step25999.pt