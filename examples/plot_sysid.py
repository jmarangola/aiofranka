"""Plot qpos vs qdes from sysid data to verify tracking quality."""

import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path to .npz file from 01_collect_ref_traj.py")
args = parser.parse_args()

data = np.load(args.path)
qpos = data["qpos"]
qdes = data["qdes"]
time = data["time"]

num_joints = qpos.shape[1]
fig, axes = plt.subplots(num_joints, 1, figsize=(14, 2.5 * num_joints), sharex=True)

for j in range(num_joints):
    ax = axes[j]
    ax.plot(time, qdes[:, j], label="qdes", linewidth=1.5, alpha=0.8)
    ax.plot(time, qpos[:, j], label="qpos", linewidth=1.5, alpha=0.8)
    ax.set_ylabel(f"Joint {j+1} (rad)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    tracking_err = np.abs(qpos[:, j] - qdes[:, j])
    ax.set_title(
        f"Joint {j+1}  |  mean err: {tracking_err.mean():.4f} rad  "
        f"max err: {tracking_err.max():.4f} rad",
        fontsize=10,
    )

axes[-1].set_xlabel("Time (s)")
fig.suptitle(args.path.split("/")[-1], fontsize=12, y=1.0)
plt.tight_layout()

out_path = args.path.replace(".npz", "_tracking.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.show()
