#!/bin/bash
# QB: first-version recipe, 4-band 530K model, 500 epochs; evaluate 5 tail checkpoints.
ROOT=/root/autodl-tmp/S2W5090
CODE=$ROOT/code_4b
OUT_ARG=${1:-runs/qb_530k_s1}
case "$OUT_ARG" in /*) OUT=$OUT_ARG ;; *) OUT=$ROOT/$OUT_ARG ;; esac
DATA=$ROOT/data
TEST=$DATA/test_qb_multiExm1.h5
VAL=$DATA/valid_qb_slim.h5
PY=/root/miniconda3/bin/python
MODEL_ARGS="--arch fm4 --c-ms 4 --c-conv 32 --agg fm --agg-d 64"

mkdir -p $OUT
cd $CODE || exit 1
echo "[$(date)] start" >> $OUT/pipeline.log

$PY train.py --data $DATA/train_qb_slim.h5 --out $OUT $MODEL_ARGS \
  --mamba-init gaussian --optim adamw --lr 4e-4 --sched step \
  --step-size 100 --gamma 0.7 --batch-size 32 --epochs 500 --seed 1 \
  --aug none --gpu-data 1 --save-every 20 \
  >> $OUT/train.log 2>&1
echo "[$(date)] training exit=$?" >> $OUT/pipeline.log

for ep in 420 440 460 480 500; do
  $PY eval_valid.py --valid $VAL $MODEL_ARGS --ckpt-glob "$OUT/ep$ep.pt" > $OUT/val_$ep.txt 2>&1
  rm -rf $ROOT/runs/test_qb_$ep
  $PY test_fm.py --file_path $TEST --save_dir $ROOT/runs/test_qb_$ep \
    --weight $OUT/ep$ep.pt --cut_size 256 --pad 0 $MODEL_ARGS > $OUT/test_$ep.log 2>&1
  $PY $ROOT/eval_psnr.py $TEST $ROOT/runs/test_qb_$ep 2047 30 > $OUT/test_${ep}_psnr.txt 2>&1
  $PY $ROOT/paired_eval.py $TEST $ROOT/runs/test_qb_$ep $ROOT/cannet_qb_rr.json > $OUT/test_${ep}_paired.txt 2>&1
done

{
  echo "=== QB: step 4e-4/100/0.7, 4-band C_conv=32 D=64, 500 ep, seed 1 ==="
  echo "--- train tail ---"; tail -3 $OUT/train.log
  for ep in 420 440 460 480 500; do
    echo "--- ep$ep val ---"; tail -1 $OUT/val_$ep.txt
    echo "--- ep$ep test ---"; cat $OUT/test_${ep}_psnr.txt; cat $OUT/test_${ep}_paired.txt
  done
} > $OUT/SUMMARY.txt
echo "[$(date)] done" >> $OUT/pipeline.log
