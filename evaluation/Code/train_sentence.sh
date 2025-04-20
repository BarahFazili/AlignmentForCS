# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
TASK=${1:-Sentiment_EN_ES}
MODEL=${2:-bert-base-multilingual-cased}
MODEL_TYPE=${3:-bert}
DATA_DIR=${4:-"$REPO/Data/Processed_Data"}
OUT_DIR=${5:-"$REPO/Results"}

EPOCH=10
BATCH_SIZE=16
MAX_SEQ=256 #256
LEARNING_RATE=1.5e-5 #1e-4   1e-3

dir=`basename "$TASK"`
if [ $dir == "Devanagari" ] || [ $dir == "Romanized" ]; then
  OUT=`dirname "$TASK"`
else
  OUT=$TASK
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
python $PWD/Code/BertSequence.py \
  --data_dir $DATA_DIR/$TASK \
  --output_dir "./oct-5_models/muse_cont_Deva/SA-51" \
  --model_type $MODEL_TYPE \
  --model_name $MODEL \
  --num_train_epochs $EPOCH \
  --train_batch_size $BATCH_SIZE \
  --max_seq_length $MAX_SEQ  \
  --seed 51 \
  --learning_rate $LEARNING_RATE

# --output_dir $OUT_DIR/$OUT \