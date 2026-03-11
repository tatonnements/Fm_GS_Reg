import roma
import torch

def transform_gaussians(gaussian_model, T):
    """
    Apply a rigid transformation T (4x4) to a GaussianModel in-place.
    T is a torch.Tensor of shape (4, 4) on the same device as the model.
    No scaling — purely rotation + translation.
    """
    R = T[:3, :3]  # (3, 3)
    t = T[:3, 3]   # (3,)

    # 1. Transform positions
    gaussian_model._xyz = gaussian_model._xyz @ R.T + t

    # 2. Transform rotations (stored as unit quaternions)
    rot_matrices = roma.unitquat_to_rotmat(gaussian_model._rotation)  # (N, 3, 3)
    rot_matrices = R[None] @ rot_matrices                              # (N, 3, 3)
    gaussian_model._rotation = roma.rotmat_to_unitquat(rot_matrices)   # (N, 4)

    # 3. scaling, opacity, features_dc, features_rest stay unchanged

    return gaussian_model