import numpy as np

def calculate_camera_intrinsics(H, W, HFOV, VFOV):
    fx = (W / 2) / np.tan(np.radians(HFOV / 2)).astype(np.float32).item()
    fy = (H / 2) / np.tan(np.radians(VFOV / 2)).astype(np.float32).item()

    cx = W / 2
    cy = H / 2

    return fx, fy, cx, cy

def get_camera_matrix(fx, fy, cx, cy):
    camera_matrix = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1],
    ], dtype=np.float32)

    return camera_matrix