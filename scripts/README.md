# Launch scripts (archived)

These three scripts are the launch scripts used on the machine where the experiments
ran. Paths at the top (`/root/autodl-tmp/S2W5090/...`) are specific to that machine:
edit `ROOT`, `CODE`, `PY` before using them, or copy the `python train.py ...` lines
out of them.

| script | what it launches |
|---|---|
| `run_gf2.sh` | GF2 training (4-band model) plus the tail-checkpoint sweep |
| `run_qb.sh`  | QuickBird training plus the tail-checkpoint sweep |
| `run_fm_sched_ema.sh` | schedule / EMA probes on WorldView-3 |

## Two files these scripts call are not included

The scripts call `eval_psnr.py` and `paired_eval.py`, which lived next to the training
code on that machine and are **not** part of this repository. They were a quick
numpy scorer used during development.

The numbers reported in the paper were **not** produced by them: every metric in the
paper comes from the PanCollection MATLAB toolbox, with the exact calls listed in the
top-level README (Section 6) and in `eval/README.md`. Use those, not a re-implementation.

If you only need inference, `test_fm.py` writes the `.mat` files the toolbox reads, so
the scoring step needs nothing from these scripts.
