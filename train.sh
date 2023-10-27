python main.py --model_name_or_path "bert-base-uncased" \
                --data_folder "data/MixATIS" \
                --output_dir "atis_output" \
                --loss_coef_intent 0.3 \
                --loss_coef_slot 0.3 \
                --loss_coef_intent_scl 0.15 \
                --loss_coef_slot_scl 0.15 \
                --sd_loss_coef 0.1 \
                --tuning_metric "mean_intent_slot" \
                --max_seq_length 100 \
                --train_batch_size 16 \
                --eval_batch_size 16 \
                --num_train_epochs 10 \
                --logging_steps 100 \
                --slot_emb_dim 100 \
                --hidden_dim_ffw 300\
                --do_eval \
                --do_train \
                --use_soft_slot \
                --use_scl \
                --use_sd \
                --use_intent_context_attention \ 

                