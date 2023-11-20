dataset_names=("squad_v2" "newsqa" "hotpotqa" "squad" "duorc_p" "duorc_s" "drop" "wikihop" "boolq" "comqa" "cq")
task_names=("qa"           "qa"      "qa"        "qa"      "qa"    "qa"        "qa"    "qa"  "superglue"  "qa"  "qa" )
export TASK_NAME=qa
export DATASET_NAME=rte
# export CUDA_VISIBLE_DEVICES=0
if [ "${#dataset_names[@]}" -ne "${#task_names[@]}" ]; then
    echo "Arrays are of different lengths."
    exit 1
fi

bs=8
lr=5e-3
dropout=0.2
psl=16
epoch=30

# microsoft/deberta-base
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
  --output_dir checkpoints/$DATASET_NAME-deberta/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch  >> checkpoints/$DATASET_NAME-deberta/run.log 2>&1
  # --prefix
  # --pre_seq_len $psl \
