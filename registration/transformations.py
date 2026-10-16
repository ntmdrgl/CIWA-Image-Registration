import numpy as np
import cv2

def get_affine_matrix(angle, hx, hy, sx, sy, tx, ty, cx, cy):
    if (cx is None or cy is None):
        raise ValueError("Camera matrix focal center (cx, cy) not configured")
    
    # convert degrees to radians
    angle = np.radians(angle)

    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ], dtype=np.float32)

    H = np.array([
        [1,  hx, 0],
        [hy, 1,  0],
        [0,  0,  1]
    ], dtype=np.float32)

    S = np.array([
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1]
    ], dtype=np.float32)

    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1 ]
    ], dtype=np.float32)

    to_origin = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1],
    ], dtype=np.float32)

    from_origin = np.array([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0, 1],
    ], dtype=np.float32)

    affine_matrix = T @ from_origin @ R @ H @ S @ to_origin
    return affine_matrix