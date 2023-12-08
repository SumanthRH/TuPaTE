dataset_names=("newsqa")
task_names=("qa")
export CUDA_VISIBLE_DEVICES=1

if [ "${#dataset_names[@]}" -ne "${#task_names[@]}" ]; then
    echo "Arrays are of different lengths."
    exit 1
fi


bs=32
lr=1e-5
dropout=0.1
# psl=20
epoch=10

for i in "${!dataset_names[@]}"; do
  dataset=${dataset_names[$i]}
  task_name=${task_names[$i]}
  # microsoft/deberta-base
  checkpoint_dir=/data3/sumanthrh/checkpoints/${dataset}-${lr}-deberta
  mkdir -p $checkpoint_dir
  python3 run.py \
    --model_name_or_path microsoft/deberta-base  \
    --task_name $task_name \
    --dataset_name $dataset \
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
    --load_best_model_at_end  \
    --metric_for_best_model "eval_f1" | tee ${checkpoint_dir}/run.log 
  done
  # --prefix
  # --pre_seq_len $psl \
