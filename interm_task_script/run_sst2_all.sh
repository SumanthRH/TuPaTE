#!/bin/bash

# Source the JSON processing script
source interm_task_script/process_ckpts.sh

export CUDA_VISIBLE_DEVICES=2
task_name="glue"
dataset_name="sst2"
bs=32
lr=1e-5
dropout=0.1
# psl=20
epoch=10
# microsoft/deberta-base
run_rte_for_interm(){
  local interm_name=$1
  local interm_dir=$2
  if [[ "$interm_name" == "$dataset_name" ]]; then
    echo "Skipping $dataset_name"
    return
  fi
  checkpoint_dir=/data3/sumanthrh/checkpoints/target-$dataset_name-interm-$interm_name-$lr-deberta
  mkdir -p $checkpoint_dir
  python3 run.py \
    --model_name_or_path $interm_dir \
    --task_name $task_name \
    --dataset_name $dataset_name \
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
}

process_json "ckpts.json" run_rte_for_interm
