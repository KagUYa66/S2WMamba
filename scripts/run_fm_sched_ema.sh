#!/bin/bash
# FusionMamba's schedule: Adam, lr 5e-4, StepLR(200, 0.5), 500 epochs.
# Keeps raw checkpoints every 10 epochs and an EMA copy; evaluates raw/EMA/avg.
ROOT=/root/autodl-tmp/S2W5090
CODE=$ROOT/code_new
OUT_ARG=${1:-runs/fm_sched_ema_s2}
case "$OUT_ARG" in /*) OUT=$OUT_ARG ;; *) OUT=$ROOT/$OUT_ARG ;; esac
DATA=$ROOT/data
TEST=$DATA/test_wv3_multiExm1.h5
VAL=$DATA/valid_wv3.h5
PY=/root/miniconda3/bin/python

mkdir -p $OUT
cd $CODE || exit 1
echo "[$(date)] start" >> $OUT/pipeline.log

$PY train.py \
  --data $DATA/train_wv3_slim.h5 --out $OUT \
  --agg fm --agg-d 84 --mamba-init gaussian \
  --optim adam --lr 5e-4 --sched step --step-size 200 --gamma 0.5 \
  --ema 0.9999 --batch-size 32 --epochs 500 --seed 1 \
  --aug none --gpu-data 1 --save-every 10 \
  >> $OUT/train.log 2>&1
echo "[$(date)] training exit=$?" >> $OUT/pipeline.log

# post-hoc average of the final LR stage (epochs 401-500)
$PY avg_ckpt.py --out $OUT/avg10.pt \
  --ckpts $OUT/ep410.pt $OUT/ep420.pt $OUT/ep430.pt $OUT/ep440.pt $OUT/ep450.pt \
          $OUT/ep460.pt $OUT/ep470.pt $OUT/ep480.pt $OUT/ep490.pt $OUT/ep500.pt \
  > $OUT/avg10_build.log 2>&1

$PY eval_valid.py --valid $VAL --ckpt-glob "$OUT/ema.pt"   > $OUT/val_ema.txt 2>&1
$PY eval_valid.py --valid $VAL --ckpt-glob "$OUT/ep500.pt" > $OUT/val_raw500.txt 2>&1
$PY eval_valid.py --valid $VAL --ckpt-glob "$OUT/ep420.pt" > $OUT/val_raw420.txt 2>&1
$PY eval_valid.py --valid $VAL --ckpt-glob "$OUT/avg10.pt" > $OUT/val_avg10.txt 2>&1

run_test () {
  name=$1; weight=$2
  rm -rf $ROOT/runs/test_${name}
  $PY test_fm.py --file_path $TEST --save_dir $ROOT/runs/test_${name} \
    --weight $weight --cut_size 256 --agg fm --agg-d 84 > $OUT/${name}.log 2>&1
  $PY $ROOT/eval_psnr.py $TEST $ROOT/runs/test_${name} 2047 30 > $OUT/${name}_psnr.txt 2>&1
  $PY $ROOT/paired_eval.py $TEST $ROOT/runs/test_${name} $ROOT/fusionmamba_wv3_rr.json > $OUT/${name}_paired.txt 2>&1
}

run_test fm_ema    $OUT/ema.pt
run_test fm_raw500 $OUT/ep500.pt
run_test fm_raw420 $OUT/ep420.pt
run_test fm_avg10  $OUT/avg10.pt

{
  echo "=== FusionMamba schedule + EMA, gaussian init, seed 1 ==="
  echo "--- train tail ---"; tail -3 $OUT/train.log
  echo "--- val EMA ---"; tail -1 $OUT/val_ema.txt
  echo "--- val raw ep500 ---"; tail -1 $OUT/val_raw500.txt
  echo "--- val raw ep420 ---"; tail -1 $OUT/val_raw420.txt
  echo "--- val avg10 ---"; tail -1 $OUT/val_avg10.txt
  echo "--- test EMA ---"; cat $OUT/fm_ema_psnr.txt; cat $OUT/fm_ema_paired.txt
  echo "--- test raw ep500 ---"; cat $OUT/fm_raw500_psnr.txt
  echo "--- test raw ep420 ---"; cat $OUT/fm_raw420_psnr.txt
  echo "--- test avg10 ---"; cat $OUT/fm_avg10_psnr.txt; cat $OUT/fm_avg10_paired.txt
} > $OUT/SUMMARY.txt
echo "[$(date)] done" >> $OUT/pipeline.log
