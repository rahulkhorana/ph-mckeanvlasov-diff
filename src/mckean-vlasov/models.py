import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Any, Tuple
from flax.linen import attention as nn_attn


class ResBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        h = nn.swish(h)
        h = nn.Conv(self.features, (3, 3), padding="SAME")(h)
        h = nn.LayerNorm()(h)
        h = nn.swish(h)
        h = nn.Conv(self.features, (3, 3), padding="SAME")(h)
        if x.shape[-1] != self.features:
            x = nn.Conv(self.features, (1, 1))(x)
        return x + h


class TinyUNet(nn.Module):
    ch: int = 64

    def _inject(self, h, te):
        """
        h: (..., H, W, C)
        te: (B, D)   (B is the leading batch dim of h)
        Broadcast a per-sample time embedding across spatial dims.
        """
        # last dim is channels always
        C = h.shape[-1]
        B = te.shape[0]
        te_proj = nn.Dense(C)(te)  # (B, C)

        # Build a shape like (B, 1, 1, C) for any rank >= 3
        # If h has rank R, we want (B, 1, ..., 1, C) with (R-2) ones in the middle
        expand = (B,) + (1,) * (h.ndim - 2) + (C,)
        te_b = jnp.reshape(te_proj, expand)  # broadcastable to h
        return h + te_b

    def _match_hw(self, x, ref):
        """Center-crop/pad x to match ref's spatial shape on the last two axes (H,W)."""
        H_ref, W_ref = int(ref.shape[-3]), int(ref.shape[-2])  # NHWC -> indices -3,-2
        H, W = int(x.shape[-3]), int(x.shape[-2])

        # crop if larger
        if H > H_ref:
            dH = H - H_ref
            top = dH // 2
            x = x[..., top : top + H_ref, :, :]
        if W > W_ref:
            dW = W - W_ref
            left = dW // 2
            x = x[..., :, left : left + W_ref, :]

        # pad if smaller
        H, W = int(x.shape[-3]), int(x.shape[-2])
        if H < H_ref or W < W_ref:
            pad_h = max(0, H_ref - H)
            pad_w = max(0, W_ref - W)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pads = ((0, 0),)  # batch (or leading) dims – will expand later
            # Build full pad spec for arbitrary rank NHWC:
            # leading dims (if any): zeros
            lead = [(0, 0)] * (x.ndim - 4)
            pad_spec = (
                *lead,
                (pad_top, pad_bottom),  # H
                (pad_left, pad_right),  # W
                (0, 0),  # C
            )
            x = jnp.pad(x, pad_spec, mode="constant")
        return x

    @nn.compact
    def __call__(self, x, t_embed):
        # Optional sanity: enforce NHWC rank>=4
        # assert x.ndim >= 4, f"UNet expects NHWC-ish input, got shape {x.shape}"

        te = nn.relu(nn.Dense(self.ch)(t_embed))
        te = nn.relu(nn.Dense(self.ch)(te))

        # Down
        h1 = ResBlock(self.ch)(x)
        h1 = self._inject(h1, te)
        d1 = nn.max_pool(h1, window_shape=(2, 2), strides=(2, 2), padding="SAME")

        h2 = ResBlock(self.ch * 2)(d1)
        h2 = self._inject(h2, te)
        d2 = nn.max_pool(h2, window_shape=(2, 2), strides=(2, 2), padding="SAME")

        h3 = ResBlock(self.ch * 4)(d2)
        h3 = self._inject(h3, te)

        # Up
        u2 = nn.ConvTranspose(self.ch * 2, (2, 2), strides=(2, 2), padding="SAME")(h3)
        u2 = self._match_hw(u2, h2)
        u2 = jnp.concatenate([u2, h2], axis=-1)
        u2 = ResBlock(self.ch * 2)(u2)

        u1 = nn.ConvTranspose(self.ch, (2, 2), strides=(2, 2), padding="SAME")(u2)
        u1 = self._match_hw(u1, h1)
        u1 = jnp.concatenate([u1, h1], axis=-1)
        u1 = ResBlock(self.ch)(u1)

        eps = nn.Conv(x.shape[-1], (1, 1), padding="SAME")(u1)
        return eps


def _robust_stats(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr).astype(np.float32)
    finite = np.isfinite(a)
    fr = float(finite.mean()) if a.size else 0.0
    a = np.where(finite, a, np.nan)
    if a.size == 0 or fr == 0.0:
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, fr, 0, 0], dtype=np.float32)
    mean = np.nanmean(a)
    std = np.nanstd(a)
    q25, q50, q75 = np.nanpercentile(a, [25, 50, 75])
    mn, mx = np.nanmin(a), np.nanmax(a)
    l1 = np.nansum(np.abs(a))
    l2 = float(np.sqrt(np.nansum(a * a)))
    max_abs = float(np.nanmax(np.abs(a)))
    numel = float(a.size)
    log_numel = float(np.log1p(numel))
    centered = a - mean
    m3 = np.nanmean(centered**3)
    skew = float(m3 / (std**3 + 1e-6))
    return np.array(
        [mean, std, mn, mx, q25, q50, q75, l1, l2, log_numel, fr, max_abs, skew],
        dtype=np.float32,
    )


def featurize_modules_trajectory(
    mods_trajectory: List[
        List[Any]
    ],  # length T; each is a list of tensors (the set at step t)
    T_max: int = 8,
    S_max: int = 16,
    add_pos_ids: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns:
      feats: (T_max, S_max, F) float32
      set_mask: (T_max, S_max, 1) 1=valid element in set
      time_mask: (T_max, 1)        1=valid timestep
    We truncate/pad both time and set sizes.
    """
    traj = mods_trajectory[:T_max]
    T = len(traj)

    feat_slices = []
    set_masks = []
    for t in range(T_max):
        if t < T:
            elems = traj[t][:S_max]
            row = []
            for i, e in enumerate(elems):
                f = _robust_stats(np.array(e))
                if add_pos_ids:  # element position in set
                    oh = np.zeros((S_max,), dtype=np.float32)
                    oh[min(i, S_max - 1)] = 1.0
                    f = np.concatenate([f, oh], axis=0)
                row.append(f)
            if len(row) == 0:
                F = 13 + (S_max if add_pos_ids else 0)
                row = [np.zeros((F,), dtype=np.float32)]
            F = row[0].shape[0]
            if len(row) < S_max:
                row += [np.zeros((F,), dtype=np.float32)] * (S_max - len(row))
            feat_slices.append(np.stack(row, axis=0))
            set_masks.append(
                np.array(
                    [1.0] * min(len(traj[t]), S_max)
                    + [0.0] * (S_max - min(len(traj[t]), S_max))
                )[:, None]
            )
        else:
            F = (
                feat_slices[0].shape[1]
                if feat_slices
                else (13 + (S_max if add_pos_ids else 0))
            )
            feat_slices.append(np.zeros((S_max, F), dtype=np.float32))
            set_masks.append(np.zeros((S_max, 1), dtype=np.float32))

    feats = jnp.array(np.stack(feat_slices, axis=0))  # (T_max,S_max,F)
    set_mask = jnp.array(np.stack(set_masks, axis=0))  # (T_max,S_max,1)
    time_mask = jnp.array(np.array([1.0] * T + [0.0] * (T_max - T), dtype=np.float32))[
        :, None
    ]  # (T_max,1)
    return feats, set_mask, time_mask


class MHA(nn.Module):
    d: int
    h: int = 4

    @nn.compact
    def __call__(self, qx, kx=None, vx=None, q_mask=None, k_mask=None):
        """
        qx: (B, Q, d)  kx,vx: (B, K, d) or None -> self-attention if None
        q_mask: (B, Q) bool   k_mask: (B, K) bool
        """
        if kx is None:
            kx = qx
        if vx is None:
            vx = kx

        q = nn.Dense(self.d)(qx)
        k = nn.Dense(self.d)(kx)
        v = nn.Dense(self.d)(vx)

        attn_mask = None
        if (q_mask is not None) and (k_mask is not None):
            # shape (B, 1, Q, K), broadcastable to heads
            attn_mask = nn_attn.make_attention_mask(q_mask, k_mask)

        return nn.MultiHeadDotProductAttention(num_heads=self.h)(
            q, k, v, mask=attn_mask
        )


class SAB(nn.Module):
    d: int
    h: int = 4

    @nn.compact
    def __call__(self, x, key_mask=None):
        """
        x: (B, S, d)
        key_mask: (B, S, 1) or (B, S) -> True keeps token
        """
        B, S, _ = x.shape
        k_mask_bool = None
        if key_mask is not None:
            k_mask_bool = (
                (key_mask[..., 0] > 0) if key_mask.ndim == 3 else key_mask.astype(bool)
            )
        # self-attn: q_mask = all ones (valid) over S
        q_mask_bool = jnp.ones((B, S), dtype=bool)
        y = MHA(self.d, self.h)(x, None, None, q_mask=q_mask_bool, k_mask=k_mask_bool)
        x = x + y
        x = x + nn.Dense(self.d)(nn.gelu(nn.Dense(self.d * 4)(x)))
        return x


class PMA(nn.Module):
    d: int
    m: int = 1
    h: int = 4

    @nn.compact
    def __call__(self, x, key_mask=None):
        """
        x: (B, S, d)
        key_mask: (B, S, 1) or (B, S) bool-like
        returns: (B, m, d)
        """
        B, S, d = x.shape
        seeds = self.param(
            "seed", nn.initializers.normal(stddev=0.02), (self.m, self.d)
        )
        q = jnp.repeat(seeds[None, :, :], B, axis=0)  # (B, m, d)

        k_mask_bool = None
        if key_mask is not None:
            k_mask_bool = (
                (key_mask[..., 0] > 0) if key_mask.ndim == 3 else key_mask.astype(bool)
            )
        q_mask_bool = jnp.ones((B, self.m), dtype=bool)

        return MHA(self.d, self.h)(q, x, x, q_mask=q_mask_bool, k_mask=k_mask_bool)


def embed_modules_trajectory(enc_apply, enc_params, mods_traj_batch, T_max=8, S_max=16):
    feats_list, set_masks, time_masks = [], [], []
    for traj in mods_traj_batch:  # batch length B
        feats, s_mask, t_mask = featurize_modules_trajectory(
            traj, T_max=T_max, S_max=S_max
        )
        feats_list.append(feats)
        set_masks.append(s_mask)
        time_masks.append(t_mask)
    feats_b = jnp.stack(feats_list, axis=0)  # (B,T,S,F)
    set_b = jnp.stack(set_masks, axis=0)  # (B,T,S,1)
    time_b = jnp.stack(time_masks, axis=0)  # (B,T,1)
    return enc_apply({"params": enc_params}, feats_b, set_b, time_b)  # (B,out)


class ModulesTrajectoryEncoder(nn.Module):
    d_set: int = 128  # per-set emb dim
    d_time: int = 256  # temporal model dim
    out_dim: int = 256
    n_sab: int = 2
    n_layers_time: int = 2
    heads: int = 4
    m_pma: int = 1
    dropout_rate: float = 0.0
    deterministic: bool = True  # set False during train if you enable dropout

    @nn.compact
    def __call__(
        self, feats: jnp.ndarray, set_mask: jnp.ndarray, time_mask: jnp.ndarray
    ):
        """
        feats:     (B,T,S,F)
        set_mask:  (B,T,S,1)  float/bool; 1=valid
        time_mask: (B,T,1)    float/bool; 1=valid
        returns:   (B,out_dim)
        """
        B, T, S, F = feats.shape

        # -------- per-time-step set encoder (SAB + PMA) --------
        x = feats.reshape(B * T, S, F)  # (B*T, S, F)
        m = set_mask.reshape(B * T, S, 1)  # (B*T, S, 1)

        # project to set dim
        x = nn.Dense(self.d_set)(x)  # (B*T, S, d_set)

        # SAB stack (self-attn over set elements, mask-aware)
        for _ in range(self.n_sab):
            x = SAB(self.d_set, self.heads)(x, key_mask=m)  # (B*T, S, d_set)

        # PMA pooled representation per time step
        x = PMA(self.d_set, m=self.m_pma, h=self.heads)(
            x, key_mask=m
        )  # (B*T, m, d_set)
        x = x.squeeze(1)  # (B*T, d_set)

        # back to (B, T, d_set)
        x = x.reshape(B, T, self.d_set)

        # -------- temporal encoder (Transformer over T) --------
        # project to temporal model dim
        x = nn.Dense(self.d_time)(x)  # (B, T, d_time)

        # boolean time mask (B,T)
        tmask_bool = (
            (time_mask.squeeze(-1) > 0)
            if time_mask.ndim == 3
            else time_mask.astype(bool)
        )
        # SelfAttention expects (B, 1, T, T) mask
        attn_mask = nn_attn.make_attention_mask(tmask_bool, tmask_bool)  # (B,1,T,T)

        for _ in range(self.n_layers_time):
            # pre-norm block
            h = nn.LayerNorm()(x)
            y = nn.SelfAttention(num_heads=self.heads, qkv_features=self.d_time)(
                h, mask=attn_mask
            )  # (B, T, d_time)
            if self.dropout_rate > 0:
                y = nn.Dropout(self.dropout_rate)(y, deterministic=self.deterministic)
            x = x + y

            h = nn.LayerNorm()(x)
            y = nn.Dense(self.d_time * 4)(h)
            y = nn.gelu(y)
            y = nn.Dense(self.d_time)(y)
            if self.dropout_rate > 0:
                y = nn.Dropout(self.dropout_rate)(y, deterministic=self.deterministic)
            x = x + y  # (B, T, d_time)

        # masked mean over time
        time_w = time_mask  # (B,T,1)
        denom = jnp.clip(jnp.sum(time_w, axis=1), 1e-6, None)  # (B,1)
        x_mean = jnp.sum(x * time_w, axis=1) / denom  # (B, d_time)

        # head
        z = nn.Dense(self.out_dim)(
            nn.gelu(nn.Dense(self.out_dim)(x_mean))
        )  # (B,out_dim)
        return z


class EnergyNetwork(nn.Module):
    """E_phi(L, M): takes image L (B,H,W,C) and M-embedding (B,d) -> scalar per sample."""

    ch: int = 64
    m_dim: int = 256

    @nn.compact
    def __call__(self, L, m_emb):
        # Expect NHWC (B,H,W,C)
        assert L.ndim >= 4, f"EnergyNetwork expects NHWC input, got {L.shape}"

        h = ResBlock(self.ch)(L)
        h = nn.max_pool(h, (2, 2), strides=(2, 2), padding="SAME")
        h = ResBlock(self.ch * 2)(h)
        h = nn.max_pool(h, (2, 2), strides=(2, 2), padding="SAME")
        h = ResBlock(self.ch * 4)(h)  # (B, H', W', C')

        # Global average over all spatial axes (everything except batch & channels)
        spatial_axes = tuple(range(1, h.ndim - 1))  # e.g. (1, 2) for 2D images
        h = jnp.mean(h, axis=spatial_axes)  # -> (B, C')

        # Ensure m_emb is (B, d)
        assert (
            m_emb.ndim == 2 and m_emb.shape[0] == h.shape[0]
        ), f"m_emb must be (B,d), got {m_emb.shape}"

        joint = jnp.concatenate([h, m_emb], axis=-1)  # (B, C'+d)
        e = nn.Dense(256)(joint)
        e = nn.tanh(e)
        e = nn.Dense(1)(e)
        return e.squeeze(-1)  # (B,)
