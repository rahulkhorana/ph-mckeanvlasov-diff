import os
import numpy as np
import matplotlib.pyplot as plt


def unpack_landscapes(arr, KS: int):
    """
    arr: (B, steps, H, W, C) with C = 3*KS
    -> (B, steps, H, W, 3, KS)
    """
    B, S, H, W, C = arr.shape
    assert C == 3 * KS, f"Expected C=3*KS, got C={C}, KS={KS}"
    return arr.reshape(B, S, H, W, 3, KS)


def to_rgb(lans_3ks, mode="k0", k_idx=0):
    """
    lans_3ks: (B, H, W, 3, KS) final-step landscapes
    mode:
      - "k0":   use specific k index -> RGB = (H0_k, H1_k, H2_k)
      - "mean": average across KS -> RGB = (mean H0, mean H1, mean H2)
      - "pca":  project 6 chans -> 3 via PCA (fit per-batch)
    returns: (B, H, W, 3) float32
    """
    B, H, W, D, KS = lans_3ks.shape  # D=3 degrees
    if mode == "k0":
        assert 0 <= k_idx < KS, f"k_idx out of range 0..{KS-1}"
        rgb = np.stack(
            [
                lans_3ks[..., 0, k_idx],  # H0
                lans_3ks[..., 1, k_idx],  # H1
                lans_3ks[..., 2, k_idx],  # H2
            ],
            axis=-1,
        )
        return rgb.astype(np.float32)
    elif mode == "mean":
        m = lans_3ks.mean(axis=-1)  # (B,H,W,3)
        return m.astype(np.float32)
    elif mode == "pca":
        X = lans_3ks.reshape(B, H, W, D * KS).astype(np.float32)  # (B,H,W,6)
        Xc_list = []
        for b in range(B):
            Xb = X[b].reshape(-1, D * KS)
            mu = Xb.mean(axis=0, keepdims=True)
            Xc = Xb - mu
            # tiny PCA: cov in feature space
            cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)  # (6,6)
            eigvals, eigvecs = np.linalg.eigh(cov)
            Wp = eigvecs[:, -3:]  # top 3
            Y = Xc @ Wp  # (H*W, 3)
            Xc_list.append(Y.reshape(H, W, 3))
        rgb = np.stack(Xc_list, axis=0)
        return rgb
    else:
        raise ValueError(f"Unknown mode {mode}")


def normalize_to_uint8(imgs, per_image=True, clip_percentile=1.0):
    """
    imgs: (B,H,W,3) float
    Returns uint8 in [0,255]. If per_image, normalize each sample separately.
    clip_percentile: e.g. 1.0 -> clip to [1,99] percentiles before scaling.
    """
    x = imgs.copy()
    if clip_percentile is not None and clip_percentile > 0:
        lo = clip_percentile
        hi = 100.0 - clip_percentile
        if per_image:
            for i in range(x.shape[0]):
                a = np.percentile(x[i], lo)
                b = np.percentile(x[i], hi)
                if b <= a:
                    b = a + 1e-6
                x[i] = np.clip(x[i], a, b)
        else:
            a = np.percentile(x, lo)
            b = np.percentile(x, hi)
            if b <= a:
                b = a + 1e-6
            x = np.clip(x, a, b)
    # scale to [0,1]
    if per_image:
        mins = x.reshape(x.shape[0], -1, 3).min(axis=1, keepdims=True)
        maxs = x.reshape(x.shape[0], -1, 3).max(axis=1, keepdims=True)
        denom = np.clip(maxs - mins, 1e-8, None)
        x = (x - mins.reshape(-1, 1, 1, 3)) / denom.reshape(-1, 1, 1, 3)
    else:
        a = x.min()
        b = x.max()
        denom = max(b - a, 1e-8)
        x = (x - a) / denom
    return (np.clip(x, 0, 1) * 255.0).astype(np.uint8)


def psnr(x, y, max_val=1.0):
    """
    x,y: (H,W,3) or (B,H,W,3) in [0,1]
    """
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    mse = np.mean((x - y) ** 2, axis=tuple(range(x.ndim - 1)))
    return 20.0 * np.log10(max_val) - 10.0 * np.log10(np.clip(mse, 1e-12, None))


# -------------- main --------------

if __name__ == "__main__":
    KS = 2
    npy_path = "samples_landscapes.npy"  # (B, steps, H, W, 3*KS)
    out_dir = "rgb_final"
    os.makedirs(out_dir, exist_ok=True)

    arr = np.load(npy_path)  # (B, steps, H, W, C)
    B, S, H, W, C = arr.shape
    lans = unpack_landscapes(arr, KS=KS)  # (B, S, H, W, 3, KS)

    final = lans[:, -1]  # (B, H, W, 3, KS) last step
    # choose RGB mapping:
    #   mode="k0" (use k=0), "mean" (avg over k), or "pca" (6->3)
    rgb_f = to_rgb(final, mode="k0", k_idx=0)  # (B, H, W, 3) float
    # If you plan PSNR/FID vs ground-truth *RGB* landscapes (in [0,1]),
    # keep a [0,1] copy and only save PNGs as uint8 separately.
    rgb_f_unit = rgb_f.copy()
    rgb_png = normalize_to_uint8(rgb_f, per_image=True, clip_percentile=1.0)

    # save PNGs
    for i in range(B):
        plt.imsave(os.path.join(out_dir, f"sample_{i:03d}.png"), rgb_png[i])

    print(f"Saved {B} RGB images to {out_dir}/")

    # -------- optional: PSNR vs GT --------
    # If you have GT landscapes as (B,H,W,3*KS) or (B,3,KS,H,W):
    # Example loader stubs (replace with your actual GT):
    # gt = np.load("gt_landscapes.npy")  # shape must match H,W and channels
    # If GT is (B,3,KS,H,W), turn to (B,H,W,3,KS) then to RGB the same way:
    # gt_hwck = np.transpose(gt, (0, 3, 4, 1, 2))  # (B,H,W,3,KS)
    # gt_rgb  = to_rgb(gt_hwck, mode="k0", k_idx=0)
    # # Ensure both in [0,1] for PSNR
    # x = np.clip(rgb_f_unit, 0, 1)
    # y = np.clip(gt_rgb,     0, 1)
    # print("PSNR per-image:", psnr(x, y, max_val=1.0))
