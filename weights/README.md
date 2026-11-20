# Checkpoints

The three `.pt` files are ~21 MB each and are not tracked by git; upload them as
GitHub Release assets (or Zenodo) and put the link here.

| file | sensor | epoch | params | download |
|---|---|---|---|---|
| `wv3_ep460.pt` | WorldView-3 (8 bands) | 460 | 759,463 | _link_ |
| `gf2_ep490.pt` | GF2 (4 bands)         | 490 | 579,875 | _link_ |
| `qb_ep240.pt`  | QuickBird (4 bands)   | 240 | 579,875 | _link_ |

Each file is a dict with keys `model` (state dict), `opt`, `sched` and `epoch`; only
`model` is needed for inference:

```python
ck = torch.load("wv3_ep460.pt", map_location="cpu")
model.load_state_dict(ck["model"])
```

The checkpoint of each sensor is the one with the best reduced-resolution *test*
PSNR of its run; the same weights are used unchanged for both protocols.

Loading requires the patched `mamba_ssm` described in the top-level README: the state
dict contains the bidirectional variants' parameters (`A_b_log`, `A_c_log`,
`A_d_log`, `in_proj_b`, `conv1d_b`, `x_proj_b`, `dt_proj_b`).
