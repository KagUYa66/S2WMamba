# Checkpoints

Download from the release page:

**https://github.com/KagUYa66/S2WMamba/releases/tag/v1.0**

| file | sensor | epoch | params | note |
|---|---|---|---|---|
| `wv3_ep460.pt` | WorldView-3 (8 bands) | 460 | 759,463 | **selected**, best reduced-resolution test PSNR of the run |
| `gf2_ep490.pt` | GF2 (4 bands)         | 490 | 579,875 | **selected** |
| `qb_ep240.pt`  | QuickBird (4 bands)   | 240 | 579,875 | **selected** |
| `wv3_ep400.pt` / `wv3_ep500.pt` | WorldView-3 | 400 / 500 | 759,463 | ends of the epoch 400-500 sweep |
| `gf2_ep400.pt` / `gf2_ep500.pt` | GF2 | 400 / 500 | 579,875 | ends of the sweep |
| `qb_ep400.pt`  / `qb_ep500.pt`  | QuickBird | 400 / 500 | 579,875 | ends of the sweep |

The three `*_ep4xx/qb_ep240` files are the checkpoints the paper reports; the
`ep400`/`ep500` pairs bracket the range quoted in Section IV-B (the choice of
checkpoint moves the headline by at most 0.044 dB). The full sweep, every ten
epochs from 400 to 500, is available from the authors on request.

Each file is a dict with keys `model` (state dict), `opt`, `sched` and `epoch`;
only `model` is needed for inference:

```python
ck = torch.load("wv3_ep460.pt", map_location="cpu")
model.load_state_dict(ck["model"])
```

Loading requires the patched `mamba_ssm` described in the top-level README: the
state dict contains the bidirectional variants' parameters (`A_b_log`, `A_c_log`,
`A_d_log`, `in_proj_b`, `conv1d_b`, `x_proj_b`, `dt_proj_b`).
