from __future__ import annotations

import os, sys
from pathlib import Path as _ArtifactPath
_PKG_ROOT = next(p for p in _ArtifactPath(__file__).resolve().parents if (p/'reproduce.py').is_file())
for _path in ('experiments/support', 'experiments/submission_scripts'):
    sys.path.append(str(_PKG_ROOT/_path))

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

N_LAYERS, N_HEADS, HID = 24, 8, 16
VARIANTS = ("L", "V", "LV")
_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def log_r_groups(X_raw):
    """Raw context X (n, d) -> (u = log r_j per row (n,), group id per row (n,), n_groups)."""
    X = np.ascontiguousarray(np.asarray(X_raw, np.float64))
    _, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    return np.log(cnt[inv]).astype(np.float32), inv.astype(np.int64), int(len(cnt))


class Phi(nn.Module):
    """phi(u) - phi(0) with phi a 1 -> HID -> 1 tanh MLP; zero_out zero-initialises the output layer."""

    def __init__(self, zero_out: bool, gen: torch.Generator):
        super().__init__()
        self.l1 = nn.Linear(1, HID)
        self.l2 = nn.Linear(HID, 1)
        with torch.no_grad():

            self.l1.weight.copy_(torch.empty(HID, 1).uniform_(-1.0, 1.0, generator=gen))
            self.l1.bias.copy_(torch.empty(HID).uniform_(-2.0, 2.0, generator=gen))
            if zero_out:
                self.l2.weight.zero_(); self.l2.bias.zero_()
            else:
                self.l2.weight.copy_(torch.empty(1, HID).uniform_(-0.5, 0.5, generator=gen)); self.l2.bias.zero_()

    def forward(self, u):
        u = u.reshape(-1, 1)
        h = torch.tanh(self.l1(u)); h0 = torch.tanh(self.l1(torch.zeros(1, 1, device=u.device, dtype=u.dtype)))
        return (self.l2(h) - self.l2(h0)).reshape(-1)


class RepAdapter(nn.Module):
    def __init__(self, variant: str = "LV", n_layers: int = N_LAYERS, n_heads: int = N_HEADS, seed: int = 0):
        super().__init__()
        assert variant in VARIANTS, variant
        self.variant, self.n_layers, self.n_heads = variant, n_layers, n_heads
        gen = torch.Generator().manual_seed(seed)
        self.use_L, self.use_V = "L" in variant, "V" in variant
        if self.use_L:
            self.phi = nn.ModuleList([Phi(False, gen) for _ in range(n_layers)])
            self.a = nn.Parameter(torch.zeros(n_layers, n_heads))
        if self.use_V:
            self.s = nn.ModuleList([Phi(True, gen) for _ in range(n_layers)])

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def param_groups_count(self):
        d = {}
        if self.use_L:
            d["L: phi (24 MLPs)"] = sum(p.numel() for p in self.phi.parameters()); d["L: a_lh"] = self.a.numel()
        if self.use_V:
            d["V: s (24 MLPs)"] = sum(p.numel() for p in self.s.parameters())
        return d

    def bias(self, layer: int, u: torch.Tensor):
        """-> (H, N) float32 additive logit bias over keys, or None."""
        if not self.use_L:
            return None
        return self.a[layer].reshape(-1, 1) * self.phi[layer](u).reshape(1, -1)

    def vscale(self, layer: int, u: torch.Tensor):
        """-> (N,) float32 multiplier of the context values, or None."""
        if not self.use_V:
            return None
        return 1.0 + self.s[layer](u)

    def is_identity(self):
        with torch.no_grad():
            z = True
            if self.use_L:
                z = z and bool((self.a == 0).all())
            if self.use_V:
                z = z and all(bool((m.l2.weight == 0).all()) for m in self.s)
            return z

    def describe(self, u_grid=(1, 2, 4, 16, 64)):
        """Mean over layers of the learned functions at r in u_grid (for logging)."""
        dev = next(self.parameters()).device
        u = torch.log(torch.tensor([float(r) for r in u_grid], device=dev))
        out = {}
        with torch.no_grad():
            if self.use_L:
                b = torch.stack([self.bias(l, u).mean(0) for l in range(self.n_layers)])                            
                out["L_bias_headmean_layermean"] = {str(r): round(float(b[:, i].mean()), 4) for i, r in enumerate(u_grid)}
                out["L_bias_layer_absmax"] = round(float(b.abs().max()), 4)
                out["a_absmean"] = round(float(self.a.abs().mean()), 5)
            if self.use_V:
                s = torch.stack([self.vscale(l, u) for l in range(self.n_layers)])
                out["V_scale_layermean"] = {str(r): round(float(s[:, i].mean()), 4) for i, r in enumerate(u_grid)}
                out["V_scale_minmax"] = [round(float(s.min()), 4), round(float(s.max()), 4)]
        return out


def _models_of(reg_or_model):
    if isinstance(reg_or_model, torch.nn.Module):
        return [reg_or_model]
    if hasattr(reg_or_model, "models_"):
        return list(reg_or_model.models_)
    return [reg_or_model.model_]


def check_groups(Xp: torch.Tensor, gid: torch.Tensor, n_groups: int):
    """Xp (N, C) preprocessed context rows; gid (N,) raw group ids. Returns (ok, message)."""
    N, C = Xp.shape
    if N != gid.numel():
        return False, "n_rows %d != len(u) %d" % (N, gid.numel())
    Xp = Xp.float()
    ref = torch.zeros(n_groups, C, device=Xp.device, dtype=Xp.dtype).index_copy_(0, gid, Xp)                                 
    diff = Xp - ref[gid]
    same = (diff == 0) | (torch.isnan(Xp) & torch.isnan(ref[gid]))
    cols_ok = same.all(0)                                            
    if int(cols_ok.sum()) == 0:
        return False, "no preprocessed column is constant within the raw x-groups"
    sig = torch.nan_to_num(ref[:, cols_ok], nan=-12345.0)
    n_distinct = torch.unique(sig, dim=0).shape[0]
    if n_distinct != n_groups:
        return False, "group signatures not distinct: %d distinct for %d groups (cols used %d/%d)" % (n_distinct, n_groups, int(cols_ok.sum()), C)
    return True, "ok: %d rows, %d groups, %d/%d columns group-constant" % (N, n_groups, int(cols_ok.sum()), C)


class AdapterHook:
    """Context manager: run `models` with the adapter active for the context table X_ctx_raw.

    probe_layers / probe_reducer: optionally recompute (no grad, fp32) the TEST-row attention
    probs of the given ICL layers INCLUDING the adapter's logit bias, shape (M, H, N) for batch
    element 0, and hand them to reducer(layer, probs) (the AttentionProbe protocol).
    strict: raise on an alignment failure (else count it in n_align_fail and continue).
    """

    def __init__(self, reg_or_models, adapter: RepAdapter | None, X_ctx_raw, probe_layers=None, probe_reducer=None, strict=True):
        self.models = _models_of(reg_or_models)
        self.adapter = adapter
        u, gid, G = log_r_groups(X_ctx_raw)
        self.u_np, self.gid_np, self.n_groups = u, gid, G
        self.u = torch.as_tensor(u); self.gid = torch.as_tensor(gid)
        self.probe_layers = None if probe_layers is None else set(probe_layers)
        self.probe_reducer = probe_reducer
        self.strict = strict
        self._cur = None; self._handles = []
        self.n_calls = 0; self.n_applied = 0; self.n_forward = 0; self.n_align_ok = 0; self.n_align_fail = 0; self.align_msgs = []
        self.shapes = set()


    def _attend(self, layer, idx, q, k, v):
        B, M, H, D = q.shape; N, Hk = k.shape[1], k.shape[2]
        if N != self.u.numel():
            raise RuntimeError("adapter u has %d entries but the context has %d keys" % (self.u.numel(), N))
        u = self.u.to(q.device)
        self.shapes.add((tuple(q.shape), tuple(k.shape)))
        bias = scale = None
        if self.adapter is not None:
            with torch.autocast(device_type=q.device.type, enabled=False):
                bias = self.adapter.bias(layer, u)                   
                scale = self.adapter.vscale(layer, u)              
        if scale is not None:
            v = v * scale.to(v.dtype).reshape(1, N, 1, 1)
        qh, kh, vh = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        if Hk != H:
            kh = kh.repeat_interleave(H // Hk, dim=1); vh = vh.repeat_interleave(H // Hk, dim=1)
        mask = None if bias is None else bias.to(q.dtype).reshape(1, H, 1, N).expand(B, H, M, N)
        if self.probe_layers is not None and idx == 1 and layer in self.probe_layers:
            with torch.no_grad(), torch.autocast(device_type=q.device.type, enabled=False):
                logits = torch.einsum("hmd,hnd->mhn", qh[0].float(), kh[0].float()) / math.sqrt(D)
                if bias is not None:
                    logits = logits + bias.reshape(1, H, N)
                probs = torch.softmax(logits, dim=-1)
                self.probe_reducer(layer, probs)
        with sdpa_kernel(_BACKENDS):
            out = F.scaled_dot_product_attention(qh.contiguous(), kh.contiguous(), vh.contiguous(), attn_mask=mask)
        return out.permute(0, 2, 1, 3)


    def _model_pre(self, mod, args, kwargs):
        self.n_forward += 1
        x = args[0] if args else kwargs.get("x"); y = args[1] if len(args) > 1 else kwargs.get("y")
        if isinstance(x, dict): x = x["main"]
        if isinstance(y, dict): y = y["main"]
        if y is None or x is None:
            return
        num_train = y.shape[0]
        if num_train == 0:
            return
        gid = self.gid.to(x.device)
        for b in range(x.shape[1]):
            ok, msg = (False, "n_rows %d != len(u) %d" % (num_train, self.u.numel())) if num_train != self.u.numel() else check_groups(x[:num_train, b, :], gid, self.n_groups)
            if ok:
                self.n_align_ok += 1
            else:
                self.n_align_fail += 1; self.align_msgs.append(msg)
                if self.strict:
                    raise RuntimeError("row alignment check failed: " + msg)

    def __enter__(self):
        import tabpfn.architectures.tabpfn_v3 as v3
        self._v3 = v3; self._orig = v3.scaled_dot_product_attention; me = self

        def patched(q, k, v, *args, **kwargs):
            cur = me._cur
            if cur is not None and k is not None:
                layer, idx = cur; me._cur = (layer, idx + 1); me.n_calls += 1
                if kwargs.get("quantized_kv") is not None:
                    raise RuntimeError("AdapterHook does not support the quantized KV cache path")
                me.n_applied += 1
                return me._attend(layer, idx, q, k, v)
            return me._orig(q, k, v, *args, **kwargs)

        v3.scaled_dot_product_attention = patched
        for mdl in self.models:
            self._handles.append(mdl.register_forward_pre_hook(self._model_pre, with_kwargs=True))
            for i, blk in enumerate(mdl.icl_blocks):
                att = blk.icl_attention

                def pre(mod, args, kwargs, i=i):
                    me._cur = (i, 0)

                def post(mod, args, kwargs, output):
                    me._cur = None

                self._handles.append(att.register_forward_pre_hook(pre, with_kwargs=True))
                self._handles.append(att.register_forward_hook(post, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        self._v3.scaled_dot_product_attention = self._orig
        for h in self._handles:
            h.remove()
        self._handles.clear(); self._cur = None
        return False


def unchunk(reg):
    for mdl in reg.models_:
        mdl.inference_row_chunk_size = None; mdl.inference_col_chunk_size = None


def load_adapter(path, device="cuda"):
    d = torch.load(path, map_location="cpu", weights_only=False)
    ad = RepAdapter(d["variant"]); ad.load_state_dict(d["state_dict"]); ad.to(device).eval()
    for p in ad.parameters():
        p.requires_grad_(False)
    return ad


if __name__ == "__main__":
    for v in VARIANTS:
        ad = RepAdapter(v)
        print("variant %-2s  params %5d  %s  identity_at_init=%s" % (v, ad.n_params(), ad.param_groups_count(), ad.is_identity()))
        u = torch.log(torch.tensor([1.0, 2.0, 64.0]))
        for l in (0, 23):
            b, s = ad.bias(l, u), ad.vscale(l, u)
            assert b is None or bool((b == 0).all()), "L part must be zero at init"
            assert s is None or bool((s == 1).all()), "V part must be one at init"
    X = np.array([[0.1], [0.1], [0.3], [0.1], [0.7], [0.7]])
    u, g, G = log_r_groups(X); print("log_r_groups:", u.round(3), g, G)
    assert G == 3 and np.allclose(np.exp(u), [3, 3, 1, 3, 2, 2])
    ok, msg = check_groups(torch.tensor([[1.0, 5.0], [1.0, 6.0], [2.0, 7.0], [1.0, 8.0], [3.0, 9.0], [3.0, 1.0]]), torch.as_tensor(g), G); print(msg); assert ok
    ok, msg = check_groups(torch.tensor([[1.0], [2.0], [2.0], [1.0], [3.0], [3.0]]), torch.as_tensor(g), G); print(msg); assert not ok
    print("adapter.py OK")
