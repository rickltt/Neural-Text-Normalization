export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=3,4,5,6

accelerate launch decoder_ft.py \
--data_dir dataset \
--data_cache_dir t5_cache_dir \
--model_name_or_path ./models/t5-small \
--max_source_length 64 \
--max_target_length 64 \
--learning_rate 1e-4 \
--num_train_epochs 10 \
--weight_decay 1e-2 \
--output_dir decoder_output \
--per_device_train_batch_size 256 \
--per_device_eval_batch_size 256 \
--preprocessing_num_workers 8 \
--num_beams 1 \
--overwrite_cache