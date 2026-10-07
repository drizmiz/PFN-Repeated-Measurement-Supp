
import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import math
import numpy as np
import torch


class LogWeightHook:
    def __init__(self, reg, logw, which="test"):
        if which not in ("test", "both"):
            raise ValueError(which)
        self.models = list(reg.models_) if hasattr(reg, "models_") else [reg]
        self.logw = torch.as_tensor(np.asarray(logw, dtype=np.float64), dtype=torch.float32)
        self.which = which
        self._cur = None
        self._handles = []
        self.n_icl_calls = 0
        self.n_applied = 0
        self.shapes = set()

    def _weighted(self, q, k, v):
        with torch.no_grad(), torch.autocast(device_type=q.device.type, enabled=False):
            B, M, H, D = q.shape
            N, Hk = k.shape[1], k.shape[2]
            if N != self.logw.numel():
                raise RuntimeError("logw has %d entries but the context has %d keys" % (self.logw.numel(), N))
            lw = self.logw.to(q.device)
            qf, kf, vf = q.float(), k.float(), v.float()
            if Hk != H:
                kf = kf.repeat_interleave(H // Hk, dim=2)
                vf = vf.repeat_interleave(H // Hk, dim=2)
            logits = torch.einsum("bmhd,bnhd->bmhn", qf, kf) / math.sqrt(D) + lw.view(1, 1, 1, N)
            probs = torch.softmax(logits, dim=-1)
            out = torch.einsum("bmhn,bnhd->bmhd", probs, vf)
            self.shapes.add((tuple(q.shape), tuple(k.shape), tuple(out.shape)))
            return out.to(q.dtype)

    def __enter__(self):
        import tabpfn.architectures.tabpfn_v3 as v3
        self._v3 = v3
        self._orig = v3.scaled_dot_product_attention
        me = self

        def patched(q, k, v, *args, **kwargs):
            cur = me._cur
            if cur is not None:
                me._cur = cur + 1
                me.n_icl_calls += 1
                if k is not None and (cur == 1 or me.which == "both"):
                    me.n_applied += 1
                    return me._weighted(q, k, v)
            return me._orig(q, k, v, *args, **kwargs)

        v3.scaled_dot_product_attention = patched
        for mdl in self.models:
            for blk in mdl.icl_blocks:
                att = blk.icl_attention

                def pre(mod, args, kwargs):
                    me._cur = 0

                def post(mod, args, kwargs, output):
                    me._cur = None

                self._handles.append(att.register_forward_pre_hook(pre, with_kwargs=True))
                self._handles.append(att.register_forward_hook(post, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        self._v3.scaled_dot_product_attention = self._orig
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._cur = None
        return False


def unchunk(reg):
    for mdl in reg.models_:
        mdl.inference_row_chunk_size = None
        mdl.inference_col_chunk_size = None
