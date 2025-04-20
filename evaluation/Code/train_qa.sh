# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
TASK=${1:-QA_EN_HI}
MODEL=${2:-bert-base-multilingual-cased}
MODEL_TYPE=${3:-bert}
DATA_DIR=${4:-"$REPO/Data/Processed_Data"}
OUT_DIR=${5:-"$REPO/Results"}


EPOCH=30
BATCH_SIZE=4
MAX_SEQ=512
LEARNING_RATE=1.5e-5 
export CUBLAS_WORKSPACE_CONFIG=:4096:8
#python $PWD/Code/run_squad_bertonly.py \
python $PWD/Code/run_squad.py \
  --data_dir $DATA_DIR/$TASK \
  --output_dir $OUT_DIR/$TASK \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL \
  --do_train \
  --do_eval \
  --version_2_with_negative \
  --num_train_epochs $EPOCH \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --max_seq_length $MAX_SEQ \
  --overwrite_output_dir \
  --seed 51 \
  --learning_rate $LEARNING_RATE \
  --logging_steps 1 \
  --evaluate_during_training 
  