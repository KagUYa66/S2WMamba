# Run manifest

Every row below is one training run of the paper (or of an auxiliary probe).
Each directory contains `config.json` (the exact command-line configuration),
`log.csv` (per-epoch training and validation loss) and the checkpoints that were
kept.

| run | wave | variant | fuse | agg | arch | epochs | checkpoints kept | role |
|---|---|---|---|---|---|---|---|---|
| `fm84_step_seed1_500` | - | coupled | mamba | fm(84) | - | 500 | ep440, ep500 | pre-wavelet reference (no wavelet transform) |
| `gf2_530k_s1` | - | coupled | mamba | fm(64) | fm4 | 500 | ep460, ep500 | pre-lofix GF2 run |
| `gf2_lofix_500` | lofix | coupled | mamba | fm(64) | fm4 | 500 | ep490, ep500 | Table I/II row S2WMamba (GF2) |
| `qb_530k_s1` | - | coupled | mamba | fm(64) | fm4 | 500 | ep200 | pre-lofix QuickBird run |
| `qb_lofix_500` | lofix | coupled | mamba | fm(64) | fm4 | 500 | ep240 | Table I/II row S2WMamba (QuickBird) |
| `wv3_addagg_lofix_500` | lofix | coupled | mamba | add(84) | fm | 500 | ep380, ep500 | Table III, aggregator replaced by a sum |
| `wv3_convfuse_500` | - | coupled | conv | fm(84) | fm | 500 | ep490, ep500 | auxiliary: conv fusion on the fixed Haar base |
| `wv3_convfuse_lofix_500` | lofix | coupled | conv | fm(84) | fm | 500 | ep490, ep500 | Table III, Mamba replaced by convolution |
| `wv3_indep_gauss_500` | - | full | mamba | fm(84) | fm | 500 | ep390, ep500 | auxiliary: parallel branches, gaussian init |
| `wv3_indep_initdef_500` | - | full | mamba | fm(84) | fm | 500 | ep480, ep500 | auxiliary: parallel branches, default init |
| `wv3_indep_lofix_500` | lofix | full | mamba | fm(84) | fm | 500 | ep470, ep500 | auxiliary: independent (parallel) branches |
| `wv3_nospat_500` | - | no_spat | mamba | fm(84) | fm | 500 | ep370, ep500 | auxiliary: no spatial branch, fixed Haar base |
| `wv3_nospat_lofix_500` | lofix | no_spat | mamba | fm(84) | fm | 500 | ep470, ep500 | Table III, no spatial branch |
| `wv3_nospec_500` | - | no_spec | mamba | fm(84) | fm | 500 | ep460, ep500 | auxiliary: no spectral branch, fixed Haar base |
| `wv3_nospec_lofix_500` | lofix | no_spec | mamba | fm(84) | fm | 500 | ep480, ep500 | Table III, no spectral branch |
| `wv3_rev_lofix_500` | lofix | rev | mamba | fm(84) | fm | 500 | ep490, ep500 | Table III, reversed cascade |
| `wv3_waveconv_500` | conv | coupled | mamba | fm(84) | fm | 500 | ep400, ep500 | Table III, fully learned basis |
| `wv3_wavelofix_500` | lofix | coupled | mamba | fm(84) | fm | 500 | ep460, ep500 | Table I/II row S2WMamba (WorldView-3) |
| `wv3_wavelofixn_500` | lofixn | coupled | mamba | fm(84) | fm | 500 | ep360, ep500 | Table III, learned detail at unit norm |
| `wv3_wavepoly_500` | poly | coupled | mamba | fm(84) | fm | 500 | ep390, ep500 | poly-phase 2-D transform (not in Table III) |
| `wv3_wavesplit_500` | split | coupled | mamba | fm(84) | fm | 500 | ep330, ep500 | Table III, no low/high mixing (spectral) |

Common recipe unless stated otherwise: AdamW, lr 4e-4, StepLR (step 100, gamma 0.7), batch 32, 64x64 patches, seed 1, `--aug none`.
