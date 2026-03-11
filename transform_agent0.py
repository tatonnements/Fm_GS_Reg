"""
Load agent0_745.ply as a GaussianModel (SH=0), apply a random rigid
transformation, save the 4x4 matrix to data/toycase/trans_agent0_745.txt,
and save the transformed model to data/toycase/transformed_agent0_745.ply.
"""

import sys
import os

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJ_ROOT, "src"))

import numpy as np
import torch

from gaussian_model import GaussianModel
from transform_gaussian import transform_gaussians


def random_rigid_transform(seed=None):
    """Generate a random rigid transformation matrix (4x4)."""
    rng = np.random.default_rng(seed)
    # Random rotation via QR decomposition
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det = +1)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    # Random translation
    t = rng.uniform(-1.0, 1.0, size=(3,))

    T = np.eye(4)
    T[:3, :3] = Q
    T[:3, 3] = t
    return T


def main():
    ply_path  = os.path.join(PROJ_ROOT, "data", "toycase", "agent0_745.ply")
    trans_path = os.path.join(PROJ_ROOT, "data", "toycase", "trans_agent0_745.txt")
    out_path  = os.path.join(PROJ_ROOT, "data", "toycase", "transformed_agent0_745.ply")

    # Load
    print(f"Loading GaussianModel (SH=0) from: {ply_path}")
    gaussians = GaussianModel(sh_degree=0)
    gaussians.load_ply(ply_path)
    print(f"  Loaded {gaussians.get_size()} Gaussians.")

    # Random rigid transform
    T_np = random_rigid_transform()
    print("Random rigid transformation T (4x4):")
    print(T_np)

    T = torch.tensor(T_np, dtype=torch.float32, device="cuda")

    # Apply
    with torch.no_grad():
        transform_gaussians(gaussians, T)
    print("Transformation applied.")

    # Save matrix (row-major, space-separated)
    np.savetxt(trans_path, T_np, fmt="%.10f")
    print(f"Transformation matrix saved to: {trans_path}")

    # Save PLY
    gaussians.save_ply(out_path)
    print(f"Transformed model saved to: {out_path}")


if __name__ == "__main__":
    main()
