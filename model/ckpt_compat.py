# -*- coding: utf-8 -*-
"""Load checkpoints written before the per-site wavelet refactor.

The old models held one shared `dwt_2d` / `dwt_1d` module and one shared
`spectral_branch.idwt`. Each of those is now instantiated per call site, so the
buffer names differ. The buffers themselves are the fixed Haar filters (they
carry no learned information), so dropping them is lossless.

Anything that is NOT a wavelet buffer must still match exactly -- a mismatch
there means the checkpoint belongs to a different architecture and must fail
loudly rather than load silently with the wrong weights.
"""
BUFFER_SUFFIX = ('.w_ll', '.w_lh', '.w_hl', '.w_hh', '.w_l', '.w_h', '.filters')
OLD_PREFIXES = ('dwt_2d.', 'dwt_1d.', 'spectral_branch.idwt.')


def load_compat(model, sd):
    sd = {k: v for k, v in sd.items() if not k.startswith(OLD_PREFIXES)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad_missing = [k for k in missing if not k.endswith(BUFFER_SUFFIX)]
    if bad_missing or unexpected:
        raise RuntimeError(
            'checkpoint mismatch -- missing learned params: %s ; unexpected keys: %s'
            % (bad_missing, unexpected))
    return model
