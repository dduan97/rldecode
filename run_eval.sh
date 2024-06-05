model=vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr/sft__44413__1708611267
python main.py \
    --mode eval \
    --model_name $model \
    --eval_task tldr:sft \
    --eval_batch_size 16 \
    --sampling_strategy fixed-temperature:1.0 \
    --eval_split validation 

    # --tp_state_path results/TRAIN:tldr:preference_temperature-policy_n:1000_valn:500_bs:3_model:vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr/sft__44413__1708611267_tpspecs:64,512,4,True_seed:44413/model_epoch45_step15199.pt