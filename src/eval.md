eval_plots_metrics.py  — class-paired by default

Evaluate generated persistence landscapes (volumes) vs. real ones.

Key features
------------
- Loads real vols (and labels) from your packed .pt via dataloader.load_packed_pt
- Loads generated volumes from one or more *.npy files (glob)
- Renders (H, W, K, C) -> RGB for metrics/plots (mode: avgk / maxk / midk / slice:k=i)
- **Class-paired metrics by default**:
    * If --gen_labels is provided → strict class pairing.
    * Else → auto-assign fake samples to classes via nearest real class centroid
      in Inception feature space (pseudo-labels), then pair within class.
- Computes global FID/KID, per-class FID/KID, and PSNR/SSIM on class-matched pairs.
- Writes metrics.json and confusion.json (when pseudo-labeling), and plots.

Run (examples)
--------------
```python
python eval_results.py \
  --real_pt ../../datasets/unified_topological_data_v6_semifast.pt \
  --gen_glob "../cpu-result/samples_landscapes.npy" \
  --outdir gen_results_fig/ \
  --max_samples 2000 \
  --device cpu
```

Optional (if your saved .npy are still standardized):
```python
  --gen_is_standardized --mu <mu> --sigma <sigma>
```



read_back.py - sanity


Usage:

```python
python viz_generated.py --npy samples_landscapes.npy --out generated_samples
```