#!/bin/bash
# GF2: first-version recipe (step 4e-4/100/0.7, gaussian init, shared pin), 4-band model.
ROOT=/root/autodl-tmp/S2W5090
CODE=$ROOT/code_4b
OUT_ARG=${1:-runs/gf2_530k_s1}
case "$OUT_ARG" in /*) OUT=$OUT_ARG ;; *) OUT=$ROOT/$OUT_ARG ;; esac
DATA=$ROOT/data
TEST=$DATA/test_gf2_multiExm1.h5
VAL=$DATA/valid_gf2_slim.h5
PY=/root/miniconda3/bin/python
MODEL_ARGS="--arch fm4 --c-ms 4 --c-conv 32 --agg fm --agg-d 64"

mkdir -p $OUT
cd $CODE || exit 1
echo "[$(date)] start" >> $OUT/pipeline.log

$PY train.py --data $DATA/train_gf2_slim.h5 --out $OUT $MODEL_ARGS \
  --mamba-init gaussian --optim adamw --lr 4e-4 --sched step \
  --step-size 100 --gamma 0.7 --batch-size 32 --epochs 480 --seed 1 \
  --aug none --gpu-data 1 --save-every 20 \
  >> $OUT/train.log 2>&1
echo "[$(date)] training exit=$?" >> $OUT/pipeline.log

$PY avg_ckpt.py --out $OUT/avg4.pt \
  --ckpts $OUT/ep420.pt $OUT/ep440.pt $OUT/ep460.pt $OUT/ep480.pt \
  > $OUT/avg4_build.log 2>&1

$PY eval_valid.py --valid $VAL $MODEL_ARGS --ckpt-glob "$OUT/ep480.pt" > $OUT/val_raw.txt 2>&1
$PY eval_valid.py --valid $VAL $MODEL_ARGS --ckpt-glob "$OUT/avg4.pt" > $OUT/val_avg4.txt 2>&1

run_test () {
  name=$1; weight=$2
  rm -rf $ROOT/runs/test_${name}
  $PY test_fm.py --file_path $TEST --save_dir $ROOT/runs/test_${name} \
    --weight $weight --cut_size 256 --pad 0 $MODEL_ARGS > $OUT/${name}.log 2>&1
  $PY $ROOT/eval_psnr.py $TEST $ROOT/runs/test_${name} 2047 30 > $OUT/${name}_psnr.txt 2>&1
  $PY $ROOT/paired_eval.py $TEST $ROOT/runs/test_${name} $ROOT/fusionmamba_gf2_rr.json > $OUT/${name}_paired.txt 2>&1
}

run_test gf2_raw480 $OUT/ep480.pt
run_test gf2_avg4   $OUT/avg4.pt

{
  echo "=== GF2: step 4e-4/100/0.7, 4-band C_conv=32 D=64, seed 1 ==="
  echo "--- train tail ---"; tail -3 $OUT/train.log
  echo "--- val raw480 ---"; tail -1 $OUT/val_raw.txt
  echo "--- val avg4 ---"; tail -1 $OUT/val_avg4.txt
  echo "--- test raw480 ---"; cat $OUT/gf2_raw480_psnr.txt; cat $OUT/gf2_raw480_paired.txt
  echo "--- test avg4 ---"; cat $OUT/gf2_avg4_psnr.txt; cat $OUT/gf2_avg4_paired.txt
} > $OUT/SUMMARY.txt
echo "[$(date)] done" >> $OUT/pipeline.log
