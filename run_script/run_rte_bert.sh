export TASK_NAME=glue
export DATASET_NAME=snli
# export CUDA_VISIBLE_DEVICES=0

bs=40
lr=1e-5
dropout=0.1
# psl=20
epoch=1
# microsoft/deberta-base
checkpoint_dir=/data3/sumanthrh/checkpoints/$DATASET_NAME-deberta
mkdir -p $checkpoint_dir
python3 run.py \
  --model_name_or_path microsoft/deberta-base  \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --output_dir $checkpoint_dir \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --save_total_limit 1 \
  --load_best_model_at_end  | tee $checkpoint_dir/run.log
  

  # --prefix
  # --pre_seq_len $psl \
