# List of datasets
# dataset_names=("snli" "mnli" "qqp" "qnli" "sst2" "scitail" "cola" "mrpc" "rte" "wnli")
dataset_names=("wnli" "wnli" "wnli" "wnli" "wnli" "wnli" "wnli" "wnli" "wnli" "wnli")
task_names=("glue" "glue" "glue" "glue" "glue" "glue" "glue" "glue" "glue" "glue")
export CUDA_VISIBLE_DEVICES="1,2"
IFS=',' read -ra gpus <<< "${CUDA_VISIBLE_DEVICES}"

if [ "${#dataset_names[@]}" -ne "${#task_names[@]}" ]; then
    echo "Arrays are of different lengths."
    exit 1
fi

threshold=200 # Memory usage threshold in MB, adjust as needed

# Function to check the memory usage of each GPU
check_gpu_memory() {
  for gpu_id in "${gpus[@]}"; do # Assuming you have two GPUs with IDs 0 and 1
    local memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id)
    if [[ $memory_used -lt $threshold ]]; then
      echo $gpu_id
    fi
  done
}


bs=40
lr=1e-5
dropout=0.1
# psl=20
epoch=1
current_index=0

# Function to start a training job on a specific GPU
start_training() {
  local gpu_id=$1
  local dataset=${dataset_names[$current_index]}
  local task_name=${task_names[$current_index]}
  echo "Starting training on GPU $gpu_id with dataset $dataset"
  checkpoint_dir=/data3/sumanthrh/checkpoints/${dataset}-${lr}-deberta
  mkdir -p $checkpoint_dir
  CUDA_VISIBLE_DEVICES=$gpu_id python3 run.py \
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
    --load_best_model_at_end  | tee ${checkpoint_dir}/run.log 
}

# Main loop to dispatch jobs to GPUs
while [ $current_index -lt ${#dataset_names[@]} ]; do
  # Get a list of GPUs with memory usage below the threshold
  readarray -t free_gpus <<< $(check_gpu_memory)
  
  for gpu_id in "${free_gpus[@]}"; do
    # Check if GPU is currently not running a job
    if ! [[ "${current_jobs[$gpu_id]}" ]]; then
      start_training $gpu_id &
      # Increment the global dataset index after starting the job
      ((current_index++))
      # Break to avoid starting another job on this iteration
      break
    fi
  done
  
  # Sleep for a bit before checking again
  sleep 10s
done


# Wait for all background jobs to finish
wait
echo "All training jobs have completed."