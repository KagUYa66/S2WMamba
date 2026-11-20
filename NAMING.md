# Branch naming: code vs paper

The two dual-branch modules are named **the opposite way** in the code and in the
paper. Nothing about the model is wrong; only the labels disagree.

| code attribute | what it actually consumes | name used in the paper, the figures and the previous submission |
|---|---|---|
| `self.spectral_branch` | MS features as backbone, modulated by the **2-D** wavelet of PAN, reconstructed by the inverse **2-D** transform | **Spatial Branch**, output `F_sp` |
| `self.spatial_branch` | PAN features as backbone, modulated by the **1-D** channel wavelet of the cue, reconstructed by the inverse **1-D** transform | **Spectral Branch**, output `F_se` |

The rule behind the paper's choice: a branch is named after the **wavelet it uses**,
not after the modality it carries. So the branch driven by the 2-D (spatial)
transform is called the Spatial Branch even though its backbone features come from
the MS image.

## Consequences

| command-line switch | removes | PSNR on the lofix base |
|---|---|---|
| `--variant no_spec` | the paper's **Spatial** branch | 38.724 ( −0.779 vs full) |
| `--variant no_spat` | the paper's **Spectral** branch | 39.362 ( −0.140 vs full) |

`--variant rev` reverses the cascade: the paper's Spectral branch runs first and the
Spatial branch is conditioned on it.

## Why the names were not fixed

Renaming the attributes would change every key in every archived checkpoint
(`archive/runs/*/ep*.pt`), and every `config.json` next to them records `variant`
using the old switch names. With the submission deadline two weeks away the
reproduction chain is worth more than the tidiness. The mapping is therefore
documented here, at the top of both model files, in `HANDOFF.md §3`, and in the
project memory.

If it is ever fixed properly, do all of the following in one commit:

1. rename the attributes in `wavelet_mamba_fm.py` and `wavelet_mamba_fm4.py`;
2. add a key-remapping step to `model/ckpt_compat.py` (`spectral_branch.*` ->
   new spatial name, and the mirror);
3. rename the `--variant` choices with the old spellings kept as aliases;
4. re-score one archived run and confirm the PSNR is bit-identical before
   touching anything else.
