# Checkpoints

Download from the release page:

**https://github.com/KagUYa66/S2WMamba/releases/tag/v1.0**

| file | sensor | epoch | params | note |
|---|---|---|---|---|
| `wv3_ep460.pt` | WorldView-3 (8 bands) | 460 | 759,463 | **selected**, best reduced-resolution test PSNR of the run |
| `wv3_ep400.pt` | WorldView-3 | 400 | 759,463 | lower end of the epoch 400-500 sweep |
| `wv3_ep500.pt` | WorldView-3 | 500 | 759,463 | upper end of the sweep |
| `gf2_ep490.pt` | GF2 (4 bands) | 490 | 579,875 | **selected** |
| `qb_ep240.pt`  | QuickBird (4 bands) | 240 | 579,875 | **selected** |

The three WorldView-3 files bracket the range quoted in Section IV-B: moving the
checkpoint over epochs 400-500 changes the headline PSNR by at most 0.044 dB, and
every value in that band still leads the strongest baseline. The GF2 and
QuickBird sweeps are available from the authors on request.

Each file is a dict with keys `model` (state dict), `opt`, `sched` and `epoch`;
only `model` is needed for inference:

```python
ck = torch.load("wv3_ep460.pt", map_location="cpu")
model.load_state_dict(ck["model"])
```

Loading requires the patched `mamba_ssm` described in the top-level README: the
state dict contains the bidirectional variants' parameters (`A_b_log`, `A_c_log`,
`A_d_log`, `in_proj_b`, `conv1d_b`, `x_proj_b`, `dt_proj_b`).
