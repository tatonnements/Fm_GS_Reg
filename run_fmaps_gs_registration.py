"""
Functional Map-based registration of two 3D Gaussian submaps.

Pipeline:
1. Load two 3D Gaussian submaps (agent0, agent1)
2. Compute LBO eigenbasis for each using the Gaussian Laplacian
3. Compute spectral descriptors (HKS/WKS) and build functional map
4. Convert functional map to point-to-point correspondences
5. Estimate rigid transformation (R, t) from correspondences with RANSAC
6. Compare with ground truth
"""

import sys
import os
import numpy as np
import torch
import scipy.sparse.linalg as sla
from scipy.linalg import svd

# Add third_party paths
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJ_ROOT, "third_party", "3DGaussianLaplacian"))

import robust_laplacian
import robust_laplacian_bindings_ext as rlbe
from scene import GaussianModel
from utils.general_utils import build_scaling_rotation


# ──────────────────────────────────────────────
# 1. Compute normals from Gaussian covariance
# ──────────────────────────────────────────────
def compute_norm(gaussians):
    """
    Compute normals from 3D Gaussian covariance matrices via SVD.
    The normal is the eigenvector corresponding to the smallest singular value.
    Reimplements extensions.utils.compute_norm in pure Python.
    """
    cov3d = gaussians.get_covariance().cpu().detach().numpy()
    cov = np.zeros((cov3d.shape[0], 3, 3))
    cov[:, 0, 0] = cov3d[..., 0]
    cov[:, 0, 1] = cov3d[..., 1]
    cov[:, 0, 2] = cov3d[..., 2]
    cov[:, 1, 0] = cov3d[..., 1]
    cov[:, 1, 1] = cov3d[..., 3]
    cov[:, 1, 2] = cov3d[..., 4]
    cov[:, 2, 0] = cov3d[..., 2]
    cov[:, 2, 1] = cov3d[..., 4]
    cov[:, 2, 2] = cov3d[..., 5]
    U, S, Vt = np.linalg.svd(cov)
    # Normal = eigenvector with smallest singular value (last column of U)
    norm = U[..., 2]
    return norm.astype(np.float64), (cov, S)


# ──────────────────────────────────────────────
# 2. Graph filtration: keep largest connected component
# ──────────────────────────────────────────────
def bfs_connected_components(neighs):
    """BFS to find connected components."""
    n_pts = len(neighs)
    visited = np.zeros(n_pts, dtype=bool)
    cc = np.zeros(n_pts, dtype=int)
    n_components = 0
    for i in range(n_pts):
        if visited[i]:
            continue
        queue = [i]
        while queue:
            src = queue.pop(0)
            if visited[src]:
                continue
            visited[src] = True
            cc[src] = n_components
            for neigh in neighs[src]:
                if not visited[neigh]:
                    queue.append(neigh)
        n_components += 1
    return cc


def graph_filtration(gaussians, radius_neigh=80, n_neigh=10):
    """Filter 3DGS and keep the largest connected component."""
    points = gaussians.get_xyz.cpu().detach().numpy().astype(np.float64)
    norms, (covs, S) = compute_norm(gaussians)
    RT = build_scaling_rotation(gaussians.get_scaling, gaussians._rotation).detach().cpu().numpy()
    RT_inverse = np.linalg.inv(RT).astype(np.float64).reshape(-1, 9)
    RT_inverse = RT_inverse / np.max(np.abs(RT_inverse), axis=1, keepdims=True)

    neighbors = rlbe.neighborhoodMahalanobis_bilateral(
        points, norms, RT_inverse, 1e-5, radius_neigh, n_neigh
    )
    n_pts = neighbors.shape[0]
    neighs = []
    for i in range(n_pts):
        if neighbors[i][-1] == n_pts:
            n_neighbor = np.where(neighbors[i] == n_pts)[0][0]
            neighs.append(neighbors[i][:n_neighbor])
        else:
            neighs.append(neighbors[i])

    cc = bfs_connected_components(neighs)
    max_component = 0
    max_elements = 0
    for i in range(np.max(cc) + 1):
        count = np.sum(cc == i)
        if count > max_elements:
            max_elements = count
            max_component = i
    index = np.where(cc == max_component)[0]
    print(f"  Graph filtration: {len(index)}/{n_pts} points in largest component")
    return index


# ──────────────────────────────────────────────
# 3. Compute Gaussian Laplacian with Mahalanobis distance
# ──────────────────────────────────────────────
def compute_gaussian_laplacian(gaussians, index, n_eigs=100, n_neighbors=30):
    """Compute LBO eigenbasis for a Gaussian splat using Mahalanobis distance."""
    points = gaussians.get_xyz.cpu().detach().numpy().astype(np.float64)
    norms, (covs, S) = compute_norm(gaussians)

    RT = build_scaling_rotation(gaussians.get_scaling, gaussians._rotation).detach().cpu().numpy()
    RT_inverse = np.linalg.inv(RT).astype(np.float64).reshape(-1, 9)

    L, M = rlbe.buildGaussianLaplacian_mahalanobis2(
        points[index], norms[index], RT_inverse[index], 1e-5, 80, n_neighbors, True
    )

    evals, evecs = sla.eigsh(L, n_eigs, M, sigma=1e-8)
    return evals, evecs, M, L


# ──────────────────────────────────────────────
# 4. Compute spectral descriptors (HKS + WKS)
# ──────────────────────────────────────────────
def compute_descriptors(evals, evecs, num_T=100, num_E=100):
    """Compute HKS and WKS descriptors from the eigenbasis."""
    from pyFM.signatures.HKS_functions import auto_HKS
    from pyFM.signatures.WKS_functions import auto_WKS

    hks = auto_HKS(evals, evecs, num_T, scaled=True)   # (N, num_T)
    wks = auto_WKS(evals, evecs, num_E, scaled=True)    # (N, num_E)
    # Combine descriptors
    desc = np.concatenate([hks, wks], axis=1)  # (N, num_T + num_E)
    return desc


# ──────────────────────────────────────────────
# 5. Compute functional map and correspondences
# ──────────────────────────────────────────────
def compute_functional_map(desc1, desc2, evals1, evals2, evecs1, evecs2,
                           M1, M2, n_fmap=30):
    """
    Compute the functional map C from shape 2 to shape 1 using descriptors,
    then convert to point-to-point map.
    """
    from pyFM.spectral.convert import FM_to_p2p

    k = n_fmap

    # Project descriptors into spectral domain
    # A_i = Phi_i^T @ M_i @ desc_i
    if hasattr(M1, 'toarray'):
        M1_diag = np.array(M1.diagonal())
    else:
        M1_diag = np.diag(M1)
    if hasattr(M2, 'toarray'):
        M2_diag = np.array(M2.diagonal())
    else:
        M2_diag = np.diag(M2)

    # Spectral coefficients of descriptors: (k, n_desc)
    A = evecs1[:, :k].T @ (M1_diag[:, None] * desc1)  # (k, n_desc)
    B = evecs2[:, :k].T @ (M2_diag[:, None] * desc2)  # (k, n_desc)

    # Solve for C: min ||C @ A - B||^2 + lambda * ||C @ Delta1 - Delta2 @ C||^2
    # Use the regularized formulation from fmaps_model.py
    lambda_param = 1e-3
    evals1_k = evals1[:k]
    evals2_k = evals2[:k]

    # Compute row-by-row following the fmaps_model approach
    A_At = A @ A.T  # (k, k)
    B_At = B @ A.T  # (k, k)

    C = np.zeros((k, k))
    for i in range(k):
        D_i = np.diag((evals2_k[i] - evals1_k) ** 2)
        C[i, :] = np.linalg.solve(A_At + lambda_param * D_i, B_At[i, :])

    print(f"  Functional map C shape: {C.shape}")

    # Convert FM to point-to-point correspondences
    p2p_21 = FM_to_p2p(C, evecs1[:, :k], evecs2[:, :k], use_adj=False)
    print(f"  P2P map shape: {p2p_21.shape}")
    return C, p2p_21


# ──────────────────────────────────────────────
# 6. Refine with ZoomOut
# ──────────────────────────────────────────────
def refine_zoomout(C, evecs1, evecs2, M1, M2, evals1, evals2, nit=10, step=1):
    """Refine functional map using ZoomOut iterations."""
    from pyFM.spectral.convert import FM_to_p2p, p2p_to_FM

    k2_start, k1_start = C.shape
    k1_max = evecs1.shape[1]
    k2_max = evecs2.shape[1]

    if hasattr(M2, 'toarray'):
        A2 = np.array(M2.diagonal())
    else:
        A2 = np.diag(M2)

    for it in range(nit):
        k1 = min(k1_start + step * (it + 1), k1_max)
        k2 = min(k2_start + step * (it + 1), k2_max)

        # Convert current FM to p2p
        p2p_21 = FM_to_p2p(C, evecs1[:, :C.shape[1]], evecs2[:, :C.shape[0]], use_adj=False)

        # Recompute FM at higher resolution
        C = p2p_to_FM(p2p_21, evecs1[:, :k1], evecs2[:, :k2], A2=A2)

    p2p_21 = FM_to_p2p(C, evecs1[:, :C.shape[1]], evecs2[:, :C.shape[0]], use_adj=False)
    print(f"  ZoomOut refined: C shape {C.shape}, p2p shape {p2p_21.shape}")
    return C, p2p_21


# ──────────────────────────────────────────────
# 7. Rigid transformation estimation with RANSAC
# ──────────────────────────────────────────────
def estimate_rigid_transform_svd(src, dst):
    """Estimate rigid transform (R, t) from corresponding point sets using SVD."""
    centroid_src = src.mean(axis=0)
    centroid_dst = dst.mean(axis=0)
    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst

    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_dst - R @ centroid_src
    return R, t


def ransac_rigid_transform(pts1, pts2, n_iter=5000, inlier_thresh=0.05):
    """
    RANSAC-based rigid transformation estimation.
    pts1[i] corresponds to pts2[i].
    Returns R, t such that pts2 ≈ R @ pts1 + t (transform from frame1 to frame2).
    """
    n = pts1.shape[0]
    best_inliers = 0
    best_R, best_t = np.eye(3), np.zeros(3)
    best_mask = np.zeros(n, dtype=bool)

    for _ in range(n_iter):
        # Sample 3 random correspondences
        idx = np.random.choice(n, 3, replace=False)
        src = pts1[idx]
        dst = pts2[idx]

        try:
            R, t = estimate_rigid_transform_svd(src, dst)
        except np.linalg.LinAlgError:
            continue

        # Count inliers
        transformed = (R @ pts1.T).T + t
        dists = np.linalg.norm(transformed - pts2, axis=1)
        mask = dists < inlier_thresh
        n_inliers = mask.sum()

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_R, best_t = R, t
            best_mask = mask

    # Refit on all inliers
    if best_inliers >= 3:
        best_R, best_t = estimate_rigid_transform_svd(pts1[best_mask], pts2[best_mask])

    print(f"  RANSAC: {best_inliers}/{n} inliers ({100*best_inliers/n:.1f}%)")
    return best_R, best_t, best_mask


# ──────────────────────────────────────────────
# 8. ICP refinement
# ──────────────────────────────────────────────
def icp_refine(pts1, pts2, R_init, t_init, max_iter=50, tol=1e-8, inlier_thresh=0.1):
    """Point-to-point ICP refinement starting from an initial R, t."""
    from scipy.spatial import KDTree

    R, t = R_init.copy(), t_init.copy()
    prev_error = np.inf

    for iteration in range(max_iter):
        # Transform pts1
        transformed = (R @ pts1.T).T + t

        # Find closest points in pts2
        tree = KDTree(pts2)
        dists, indices = tree.query(transformed)

        # Filter outliers
        mask = dists < inlier_thresh
        if mask.sum() < 3:
            break

        # Estimate transform from inlier correspondences
        R_new, t_new = estimate_rigid_transform_svd(pts1[mask], pts2[indices[mask]])

        R, t = R_new, t_new
        error = np.mean(dists[mask])

        if abs(prev_error - error) < tol:
            break
        prev_error = error

    print(f"  ICP converged after {iteration+1} iterations, error={error:.6f}")
    return R, t


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    np.random.seed(42)
    data_dir = os.path.join(PROJ_ROOT, "data", "toycase")
    n_eigs = 80

    # ── Step 1: Load 3D Gaussian submaps ──
    print("=" * 60)
    print("Step 1: Loading 3D Gaussian submaps")
    print("=" * 60)

    gs0 = GaussianModel(sh_degree=0)
    gs0.load_ply(os.path.join(data_dir, "agent0_745_rasterized.ply"))
    print(f"  Agent0: {gs0.get_xyz.shape[0]} Gaussians")

    gs1 = GaussianModel(sh_degree=0)
    gs1.load_ply(os.path.join(data_dir, "agent1_716_rasterized.ply"))
    print(f"  Agent1: {gs1.get_xyz.shape[0]} Gaussians")

    # ── Step 2: Graph filtration + LBO computation ──
    print("\n" + "=" * 60)
    print("Step 2: Graph filtration and LBO computation")
    print("=" * 60)

    print("  Filtering agent0...")
    idx0 = graph_filtration(gs0)
    print("  Computing LBO for agent0...")
    evals0, evecs0, M0, L0 = compute_gaussian_laplacian(gs0, idx0, n_eigs=n_eigs)
    pts0 = gs0.get_xyz.cpu().detach().numpy().astype(np.float64)[idx0]
    print(f"  Agent0 LBO: {len(evals0)} eigenvalues, evecs shape {evecs0.shape}")

    print("  Filtering agent1...")
    idx1 = graph_filtration(gs1)
    print("  Computing LBO for agent1...")
    evals1, evecs1, M1, L1 = compute_gaussian_laplacian(gs1, idx1, n_eigs=n_eigs)
    pts1 = gs1.get_xyz.cpu().detach().numpy().astype(np.float64)[idx1]
    print(f"  Agent1 LBO: {len(evals1)} eigenvalues, evecs shape {evecs1.shape}")

    # ── Step 3: Compute spectral descriptors and functional map ──
    print("\n" + "=" * 60)
    print("Step 3: Computing spectral descriptors and functional map")
    print("=" * 60)

    desc0 = compute_descriptors(evals0, evecs0, num_T=50, num_E=50)
    desc1 = compute_descriptors(evals1, evecs1, num_T=50, num_E=50)
    print(f"  Descriptors: agent0 {desc0.shape}, agent1 {desc1.shape}")

    n_fmap = 30
    C, p2p_21 = compute_functional_map(
        desc0, desc1, evals0, evals1, evecs0, evecs1, M0, M1, n_fmap=n_fmap
    )

    # ZoomOut refinement
    print("  Refining with ZoomOut...")
    C_refined, p2p_21_refined = refine_zoomout(
        C, evecs0, evecs1, M0, M1, evals0, evals1, nit=15, step=3
    )

    # ── Step 4: Convert to geometry space and estimate transformation ──
    print("\n" + "=" * 60)
    print("Step 4: Estimating rigid transformation from correspondences")
    print("=" * 60)

    # Get corresponding 3D points
    corr_pts0 = pts0[p2p_21_refined]  # points on agent0 corresponding to each agent1 point
    corr_pts1 = pts1                  # agent1 points

    # Estimate transformation: T maps agent1 -> agent0 frame
    # i.e., pts0 ≈ R @ pts1 + t
    # No ICP: Gaussians are not a dense point cloud, so nearest-neighbor
    # matching is not meaningful. Use only functional map correspondences.
    print("  Running RANSAC on functional map correspondences...")
    R_est, t_est, inlier_mask = ransac_rigid_transform(
        corr_pts1, corr_pts0, n_iter=10000, inlier_thresh=0.08
    )

    # Build 4x4 transform matrix
    T_est = np.eye(4)
    T_est[:3, :3] = R_est
    T_est[:3, 3] = t_est

    print("\n  Estimated transformation (agent1 -> agent0):")
    print(T_est)

    # ── Step 5: Compare with ground truth ──
    print("\n" + "=" * 60)
    print("Step 5: Comparison with ground truth")
    print("=" * 60)

    T_gt = np.loadtxt(os.path.join(data_dir, "rel_pose_a1f716_to_a0f745.txt"))
    print("  Ground truth transformation:")
    print(T_gt)

    # Rotation error (geodesic distance on SO(3))
    R_gt = T_gt[:3, :3]
    R_diff = R_est @ R_gt.T
    cos_angle = np.clip((np.trace(R_diff) - 1) / 2.0, -1.0, 1.0)
    rot_error_deg = np.degrees(np.arccos(cos_angle))

    # Translation error
    t_gt = T_gt[:3, 3]
    trans_error = np.linalg.norm(t_est - t_gt)

    print(f"\n  Rotation error:    {rot_error_deg:.4f} degrees")
    print(f"  Translation error: {trans_error:.6f} (Euclidean)")
    print(f"  Translation GT norm: {np.linalg.norm(t_gt):.6f}")
    print(f"  Relative trans error: {trans_error / np.linalg.norm(t_gt) * 100:.2f}%")



if __name__ == "__main__":
    main()
