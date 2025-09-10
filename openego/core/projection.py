import numpy as np
from typing import Union

def convert_points_to_trajetory_coordinates(
    points: np.ndarray, 
    intrinsic: np.ndarray,
    output_depth: bool = False
) -> Union[np.ndarray, tuple]:
    """
    Convert 3D points to 2D pixel coordinates using camera intrinsics.
    
    Args:
        points: shape [num_points, 3] or [batch_size, num_points, 3]
        intrinsic: shape [3, 3] or [batch_size, 3, 3]
        output_depth: if True, also return depth values
    
    Returns:
        pixel_coordinates: shape [..., 2] with integer pixel values
        depth (optional): shape [..., 1] if output_depth is True
    """
    # Extract principal point and focal length
    if len(intrinsic.shape) == 3:
        # Batched intrinsics
        p = intrinsic[:, :2, 2][:, np.newaxis, np.newaxis, :]
        f = np.stack([intrinsic[:, 0, 0], intrinsic[:, 1, 1]], axis=-1)[:, np.newaxis, np.newaxis, :]
    else:
        # Single intrinsic matrix
        p = np.array([intrinsic[0, 2], intrinsic[1, 2]])
        f = np.array([intrinsic[0, 0], intrinsic[1, 1]])
    
    # Extract depth
    depth = points[..., -1:].clip(min=1e-6)  # Avoid division by zero
    
    # Project to pixel coordinates
    pixel_coords = ((points[..., :2] * f) / depth) + p
    pixel_coords = pixel_coords.astype(np.int32)
    
    if output_depth:
        return pixel_coords, depth
    
    return pixel_coords