import torch
from net.st_gcn import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
graph_args = {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
x = torch.randn(2, 3, 300, 25, 2, device=device)

combos = [
    (True,  True,  "KG + DynAdj + TempAttn"),
    (True,  False, "KG + DynAdj only"),
    (False, True,  "KG + TempAttn only"),
    (False, False, "KG only (static adj + avg pool)"),
]

for use_ta, use_da, label in combos:
    model = Model(
        in_channels=3, num_class=120, graph_args=graph_args,
        edge_importance_weighting=True, use_kg=True, dropout=0.5,
        use_temporal_attention=use_ta, use_dynamic_adj=use_da,
    ).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 120), f"{label}: got {out.shape}"
    print(f"[PASS] {label} -> {out.shape}")
