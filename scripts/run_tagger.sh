export CUDA_VISIBLE_DEVICES=4
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1


python tagger_ft.py \
        --do_train \
        --do_eval \
        --do_predict \
        --output_dir output/tagger_output \
        --model_name_or_path ./models/electra-small-discriminator \
        --cache_dir cache/tagger_cache \
        --train_file dataset/train_tagger.json \
        --eval_file dataset/dev_tagger.json \
        --test_file dataset/test_tagger.json \
        --num_proc 8 \
        --max_seq_length 512 \
        --num_train_epochs 10 \
        --per_device_train_batch_size 64 \
        --per_device_eval_batch_size 64 \
        --learning_rate 3e-5 \
        --weight_decay 1e-2 \
        --logging_strategy epoch \
        --eval_strategy epoch \
        --save_strategy epoch \
        --metric_for_best_model f1 \
        --save_total_limit 2 \
        --load_best_model_at_end \
        --overwrite_output_dir \
        # --max_train_samples 1000 \
        # --max_eval_samples 100 \
        # --max_predict_samples 100 \