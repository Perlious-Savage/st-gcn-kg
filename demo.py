import matplotlib
matplotlib.use('TkAgg')

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from net.st_gcn import Model

# -------- LOAD MODEL --------
model = Model(
    in_channels=3,
    num_class=120,
    graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
    edge_importance_weighting=True
)

model.load_state_dict(torch.load(
    'work_dir/recognition/ntu-xsub/ST_GCN/epoch10_model.pt'
))

model.eval()

# -------- LOAD DATA --------
data = np.load('C:/Users/gagan/Downloads/ntu_processed/xsub/val_data.npy')
sample = torch.tensor(data[0:1]).float()

# -------- PREDICTION --------
with torch.no_grad():
    output = model(sample)
    probs = torch.softmax(output, dim=1)
    top5 = torch.topk(probs, 5)

pred_class = top5.indices[0][0].item()
confidence = top5.values[0][0].item()

print("Predicted class:", pred_class)
print("Confidence:", round(confidence*100, 2), "%")

# -------- JOINT CONNECTIONS --------
pairs = [
    (0,1),(1,2),(2,3),(3,4),
    (1,5),(5,6),(6,7),
    (1,8),(8,9),(9,10),
    (1,11),(11,12),(12,13),
    (1,14),(14,15),(15,16)
]

colors = ['black','black','black','black',
          'blue','blue','blue',
          'green','green','green',
          'red','red','red',
          'orange','orange','orange']

# -------- BODY PARTS (KG) --------
body_parts = {
    "head": [0,1,2],
    "left_arm": [5,6,7],
    "right_arm": [8,9,10],
    "torso": [1,11,12],
    "left_leg": [11,12,13],
    "right_leg": [14,15,16]
}

kg_connections = [
    ("head", "torso"),
    ("torso", "left_arm"),
    ("torso", "right_arm"),
    ("torso", "left_leg"),
    ("torso", "right_leg")
]

# -------- FIGURE (SPLIT VIEW) --------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))

# LEFT: Skeleton
sc1 = ax1.scatter([], [], s=50, color='black')
lines = []
for c in colors:
    line, = ax1.plot([], [], lw=2, color=c)
    lines.append(line)

ax1.set_title("Skeleton View")
ax1.axis('off')

# RIGHT: Knowledge Graph
sc2 = ax2.scatter([], [], s=100, color='purple')
kg_lines = []
for _ in kg_connections:
    line, = ax2.plot([], [], linestyle='--', color='purple')
    kg_lines.append(line)

ax2.set_title("Knowledge Graph View")
ax2.axis('off')

# -------- UPDATE FUNCTION --------
def update(frame_idx):
    frame = data[0][:, frame_idx, :, 0]
    x = frame[0]
    y = frame[1]

    # center + flip
    x = x - np.mean(x)
    y = -(y - np.mean(y))

    # ---- LEFT: SKELETON ----
    sc1.set_offsets(np.c_[x, y])

    for line, (i, j) in zip(lines, pairs):
        line.set_data([x[i], x[j]], [y[i], y[j]])

    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(-0.8, 0.8)
    ax1.set_title(f"Skeleton\nAction: {pred_class}")

    # ---- RIGHT: KNOWLEDGE GRAPH ----
    centers = {}
    for part, joints in body_parts.items():
        cx = np.mean([x[j] for j in joints])
        cy = np.mean([y[j] for j in joints])
        centers[part] = (cx, cy)

    # plot centers
    ax2.clear()
    ax2.axis('off')

    for part, (cx, cy) in centers.items():
        ax2.scatter(cx, cy, s=120, color='purple')
        ax2.text(cx, cy, part, fontsize=8, ha='center')

    # draw connections
    for (p1, p2), line in zip(kg_connections, kg_lines):
        x1, y1 = centers[p1]
        x2, y2 = centers[p2]
        ax2.plot([x1, x2], [y1, y2], linestyle='--', color='purple')

    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(-0.8, 0.8)
    ax2.set_title(f"Knowledge Graph\nConfidence: {round(confidence*100,1)}%")

    return [sc1] + lines

# -------- ANIMATION --------
ani = FuncAnimation(
    fig,
    update,
    frames=min(50, data.shape[2]),
    interval=100
)

plt.pause(0.001)
plt.show(block=True)