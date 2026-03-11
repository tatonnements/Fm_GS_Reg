#!/usr/bin/env python3
"""
Depth-image registration using Functional Maps.

Pipeline:
  1. Load depth images and camera intrinsics.
  2. Back-project depth images into 3D and triangulate an organized mesh.
  3. Compute LBO eigenfunctions on both meshes and build functional-map
     correspondences (pyFM) with outlier removal.
  4. Estimate a rigid transformation (R, t) from the correspondences using
     RANSAC + SVD in geometric (3-D) space.
  5. Compare with the ground-truth relative pose.
"""

import sys
import os
import argparse
import numpy as np
import open3d as o3d
import yaml
from PIL import Image

# ── make the vendored pyFM importable ──────────────────────────────────────
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "third_party", "pyFM")
)
from pyFM.mesh import TriMesh
from pyFM.functional import FunctionalMapping


# ---------------------------------------------------------------------------
# 1.  Helpers – depth image → mesh
# ---------------------------------------------------------------------------

def load_camera_config(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)["cam"]


def depth_image_to_mesh(
    depth_path: str,
    cam: dict,
    downsample: int = 4,
    max_depth_jump: float = 0.15,
):
    """
    Convert a 16-bit depth image to a triangle mesh.

    Back-projects every *downsample*-th pixel using the camera intrinsics and
    connects neighbouring valid pixels into triangles, rejecting any triangle
    whose vertices span a depth discontinuity larger than *max_depth_jump* (m).
    """
    raw = np.array(Image.open(depth_path)).astype(np.float64)
    depth = raw[::downsample, ::downsample] / cam["depth_scale"]
    h, w = depth.shape

    fx = cam["fx"] / downsample
    fy = cam["fy"] / downsample
    cx = cam["cx"] / downsample
    cy = cam["cy"] / downsample

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)          # (h, w, 3)

    valid = depth > 0                               # (h, w)

    # Sequential vertex IDs for valid pixels only
    vid = np.full((h, w), -1, dtype=np.int64)
    vid[valid] = np.arange(valid.sum())
    vertices = points[valid]                        # (N, 3)

    # ---- vectorised face generation from grid neighbours ----
    r, c = np.mgrid[: h - 1, : w - 1]
    v00, v01 = vid[r, c], vid[r, c + 1]
    v10, v11 = vid[r + 1, c], vid[r + 1, c + 1]
    d00, d01 = depth[r, c], depth[r, c + 1]
    d10, d11 = depth[r + 1, c], depth[r + 1, c + 1]

    # Triangle A: (00, 10, 01)
    mA = (v00 >= 0) & (v10 >= 0) & (v01 >= 0)
    mA &= (
        np.maximum(np.maximum(d00, d10), d01)
        - np.minimum(np.minimum(d00, d10), d01)
    ) < max_depth_jump
    tA = np.stack([v00[mA], v10[mA], v01[mA]], axis=-1)

    # Triangle B: (01, 10, 11)
    mB = (v01 >= 0) & (v10 >= 0) & (v11 >= 0)
    mB &= (
        np.maximum(np.maximum(d01, d10), d11)
        - np.minimum(np.minimum(d01, d10), d11)
    ) < max_depth_jump
    tB = np.stack([v01[mB], v10[mB], v11[mB]], axis=-1)

    faces = np.concatenate([tA, tB], axis=0)
    return vertices, faces


def decimate_mesh(vertices, faces, target_faces=30_000):
    """Optionally decimate an Open3D mesh to *target_faces* triangles."""
    if len(faces) <= target_faces:
        return vertices, faces
    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces),
    )
    mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_faces)
    mesh_o3d.remove_unreferenced_vertices()
    return np.asarray(mesh_o3d.vertices), np.asarray(mesh_o3d.triangles)


# ---------------------------------------------------------------------------
# 2.  Functional-map correspondences
# ---------------------------------------------------------------------------

def compute_fmap_correspondences(
    verts0, faces0, verts1, faces1,
    n_ev=50, n_descr=100, descr_type="WKS",
    icp_nit=10, verbose=True,
):
    """
    Build a functional map between two meshes and return the
    point-to-point vertex correspondence  p2p_21  (mesh1 → mesh0).
    """
    mesh0 = TriMesh(verts0, faces0)
    mesh1 = TriMesh(verts1, faces1)

    fm = FunctionalMapping(mesh0, mesh1)
    fm.preprocess(
        n_ev=(n_ev, n_ev),
        n_descr=n_descr,
        descr_type=descr_type,
        verbose=verbose,
    )
    fm.fit(
        w_descr=1e-1,
        w_lap=1e-3,
        w_dcomm=1,
        verbose=verbose,
    )
    p2p_21_coarse = fm.get_p2p(n_jobs=1)

    fm.icp_refine(nit=icp_nit, verbose=verbose)

    p2p_21 = fm.get_p2p(n_jobs=1)  # p2p_21[i] = index on mesh0
    return p2p_21_coarse, p2p_21, fm


# ---------------------------------------------------------------------------
# 3.  Geometric transformation estimation
# ---------------------------------------------------------------------------

def estimate_rigid_svd(src, tgt):
    """Closed-form rigid transform (R, t) so that  tgt ≈ R @ src + t ."""
    mu_s = src.mean(axis=0)
    mu_t = tgt.mean(axis=0)
    H = (src - mu_s).T @ (tgt - mu_t)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])  # correct reflection
    R = Vt.T @ D @ U.T
    t = mu_t - R @ mu_s
    return R, t


def ransac_rigid(
    src, tgt,
    n_iter=10_000,
    inlier_thresh=0.05,
    min_sample=3,
):
    """RANSAC wrapper around *estimate_rigid_svd*."""
    n = len(src)
    best_mask = np.zeros(n, dtype=bool)
    best_count = 0

    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        idx = rng.choice(n, min_sample, replace=False)
        R, t = estimate_rigid_svd(src[idx], tgt[idx])
        residuals = np.linalg.norm((R @ src.T).T + t - tgt, axis=1)
        mask = residuals < inlier_thresh
        count = mask.sum()
        if count > best_count:
            best_count = count
            best_mask = mask

    if best_count < min_sample:
        print(f"[RANSAC] Warning: only {best_count} inliers – result may be poor")

    # Refine on all inliers
    R, t = estimate_rigid_svd(src[best_mask], tgt[best_mask])
    return R, t, best_mask


# ---------------------------------------------------------------------------
# 4.  Evaluation helpers
# ---------------------------------------------------------------------------

def rotation_error_deg(R_est, R_gt):
    trace_val = np.clip((np.trace(R_est.T @ R_gt) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(trace_val))


def translation_error(t_est, t_gt):
    return np.linalg.norm(t_est - t_gt)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Depth-camera registration via Functional Maps"
    )
    parser.add_argument(
        "--data_dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "toycase"
        ),
    )
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--max_depth_jump", type=float, default=0.15)
    parser.add_argument("--target_faces", type=int, default=50_000)
    parser.add_argument("--n_ev", type=int, default=50)
    parser.add_argument("--n_descr", type=int, default=100)
    parser.add_argument("--descr_type", default="WKS", choices=["WKS", "HKS"])
    parser.add_argument("--ransac_iter", type=int, default=10_000)
    parser.add_argument("--inlier_thresh", type=float, default=0.05)
    args = parser.parse_args()

    # ── camera config ──────────────────────────────────────────────────────
    cam = load_camera_config(os.path.join(args.data_dir, "cam_config.yaml"))
    print(f"Camera: {cam['W']}×{cam['H']}, "
          f"fx={cam['fx']}, fy={cam['fy']}, depth_scale={cam['depth_scale']}")

    # ── Step 1: depth images → meshes ──────────────────────────────────────
    print("\n══ Step 1: Depth → Mesh ══")
    depth0_path = os.path.join(args.data_dir, "agent0_745_camera.png")
    depth1_path = os.path.join(args.data_dir, "agent1_716_camera.png")

    v0, f0 = depth_image_to_mesh(
        depth0_path, cam,
        downsample=args.downsample,
        max_depth_jump=args.max_depth_jump,
    )
    v1, f1 = depth_image_to_mesh(
        depth1_path, cam,
        downsample=args.downsample,
        max_depth_jump=args.max_depth_jump,
    )
    print(f"  Mesh 0  (agent0): {len(v0):>7,} verts, {len(f0):>7,} faces")
    print(f"  Mesh 1  (agent1): {len(v1):>7,} verts, {len(f1):>7,} faces")

    # optional decimation
    v0, f0 = decimate_mesh(v0, f0, target_faces=args.target_faces)
    v1, f1 = decimate_mesh(v1, f1, target_faces=args.target_faces)
    print(f"  After decimation:")
    print(f"  Mesh 0: {len(v0):>7,} verts, {len(f0):>7,} faces")
    print(f"  Mesh 1: {len(v1):>7,} verts, {len(f1):>7,} faces")

    # ── Step 2: Functional-map correspondences ─────────────────────────────
    print("\n══ Step 2: LBO + Functional Map ══")
    p2p_21_coarse, p2p_21, fm = compute_fmap_correspondences(
        v0, f0, v1, f1,
        n_ev=args.n_ev,
        n_descr=args.n_descr,
        descr_type=args.descr_type,
    )
    print(f"  p2p map (coarse): {len(p2p_21_coarse)} correspondences")
    print(f"  p2p map (refined): {len(p2p_21)} correspondences")

    # ── Step 2b: Coarse transformation (before ICP refinement) ─────────────
    print("\n══ Step 2b: Coarse Transformation (pre-ICP) ══")
    R_coarse, t_coarse, mask_coarse = ransac_rigid(
        v1, v0[p2p_21_coarse],
        n_iter=args.ransac_iter,
        inlier_thresh=args.inlier_thresh,
    )
    T_coarse = np.eye(4)
    T_coarse[:3, :3] = R_coarse
    T_coarse[:3, 3] = t_coarse
    print(f"  Inliers: {mask_coarse.sum()}/{len(v1)} "
          f"({100 * mask_coarse.sum() / len(v1):.1f}%)")
    print(f"\n  Coarse T (agent1 → agent0):\n{T_coarse}")

    # ── Step 3: Rigid transformation via RANSAC + SVD (after ICP refine) ───
    print("\n══ Step 3: Refined Transformation (post-ICP) ══")
    src_pts = v1                   # points in agent1 camera frame
    tgt_pts = v0[p2p_21]          # matched points in agent0 camera frame

    R_est, t_est, inlier_mask = ransac_rigid(
        src_pts, tgt_pts,
        n_iter=args.ransac_iter,
        inlier_thresh=args.inlier_thresh,
    )

    T_est = np.eye(4)
    T_est[:3, :3] = R_est
    T_est[:3, 3] = t_est

    n_inliers = inlier_mask.sum()
    print(f"  Inliers: {n_inliers}/{len(src_pts)} "
          f"({100 * n_inliers / len(src_pts):.1f}%)")
    print(f"\n  Refined T (agent1 → agent0):\n{T_est}")

    # ── Step 4: Compare with ground truth ──────────────────────────────────
    print("\n══ Step 4: Comparison with Ground Truth ══")
    gt_path = os.path.join(args.data_dir, "rel_pose_a1f716_to_a0f745.txt")
    T_gt = np.loadtxt(gt_path)
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]

    rot_err_coarse = rotation_error_deg(R_coarse, R_gt)
    trans_err_coarse = translation_error(t_coarse, t_gt)

    rot_err = rotation_error_deg(R_est, R_gt)
    trans_err = translation_error(t_est, t_gt)

    print(f"  Ground-truth T:\n{T_gt}")
    print(f"\n  ── Coarse (pre-ICP) ──")
    print(f"  Rotation error:    {rot_err_coarse:.4f}°")
    print(f"  Translation error: {trans_err_coarse:.6f} m")
    print(f"\n  ── Refined (post-ICP) ──")
    print(f"  Rotation error:    {rot_err:.4f}°")
    print(f"  Translation error: {trans_err:.6f} m")
    print(f"\n  Estimated  R:\n{R_est}")
    print(f"  GT         R:\n{R_gt}")
    print(f"\n  Estimated  t: {t_est}")
    print(f"  GT         t: {t_gt}")


if __name__ == "__main__":
    main()
