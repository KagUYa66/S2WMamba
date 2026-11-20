# Evaluation drivers

The metrics themselves are the PanCollection toolbox
(`Toolbox_psnr/indexes_evaluation.m`, `indexes_evaluation_FS.m`). These are the
drivers we used on top of it.

| file | what it does |
|---|---|
| `eval_awlp.m` | scores the AWLP baseline (reduced and full resolution) with the same toolbox calls as every other row |
| `eval_fr_toolbox_extra.m` | scores a set of released outputs (BDPN / MMNet / MSDCNN / MUCNN / PMAC) under the same protocol |

Drop them next to `Toolbox_psnr/` and edit the `addpath` and the data paths at the
top of each file.

Protocol reminders:

- reduced resolution: `indexes_evaluation(F, G, 4, L, 32, 1, 30, 0, 2047)` with
  `L = 11` for WorldView-3 and `L = 10` for the 4-band sensors. The border argument
  is 30, i.e. MATLAB `30:end-30`; using `31:end-30` shifts PSNR by ~0.002 dB.
- full resolution: `indexes_evaluation_FS` with a 32-pixel block size, no injection
  coefficient.
- `HQNR = (1 - D_lambda) * (1 - D_s)`, computed from the unrounded terms.
