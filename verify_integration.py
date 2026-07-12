"""
Pre-Colab verification: tests all 5 ablation configs with synthetic data.
Run this FIRST on Colab before training to catch any import/shape errors.

Usage:  python verify_integration.py
"""
import torch
from net.st_gcn import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[device] {device}')

graph_args = {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
x = torch.randn(2, 3, 300, 25, 2, device=device)

configs = [
    # (use_kg, use_temporal_attention, use_dynamic_adj, label)
    (False, None,  None,  "1. Baseline (vanilla ST-GCN)"),
    (True,  False, False, "2. KG only (static adj + avg pool)"),
    (True,  False, True,  "3. KG + Dynamic Adj only"),
    (True,  True,  False, "4. KG + Temporal Attn only"),
    (True,  True,  True,  "5. KG + Both (full)"),
]

for use_kg, use_ta, use_da, label in configs:
    kwargs = dict(
        in_channels=3, num_class=60, graph_args=graph_args,
        edge_importance_weighting=True, use_kg=use_kg, dropout=0.5,
    )
    if use_ta is not None:
        kwargs['use_temporal_attention'] = use_ta
    if use_da is not None:
        kwargs['use_dynamic_adj'] = use_da

    model = Model(**kwargs).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 60), f"{label}: expected (2, 60), got {out.shape}"
    assert torch.isfinite(out).all(), f"{label}: output contains NaN/Inf!"

    # Quick backward check
    model.train()
    out2 = model(x)
    loss = out2.sum()
    loss.backward()
    print(f"[PASS] {label} -> shape {tuple(out.shape)}, grad OK")

print("\n[ALL PASS] All 5 ablation configs verified successfully!")

# --- Label sanity check ---
import os
import pickle

label_path = '/content/drive/MyDrive/st-gcn-project/datasets/ntu_processed/xsub/train_label.pkl'
if os.path.exists(label_path):
    with open(label_path, 'rb') as f:
        names, labels = pickle.load(f)
    print(f"\n[LABEL CHECK] {label_path}")
    print(f"  Samples: {len(labels)}")
    print(f"  Label range: {min(labels)} – {max(labels)}")
    if min(labels) == 60:
        print("  ✅ Labels are 60-indexed. feeder.py dynamic remapping will handle this.")
    else:
        print(f"  ⚠️  Unexpected label range. Verify feeder.py handles this correctly.")
else:
    print(f"\n[LABEL CHECK] {label_path} not found — check path on Colab")
