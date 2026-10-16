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

# def get_camera_matrix_from_camera(camera):
#     """
#     Updates camera with calculated focal lengths and image centers
#     Returns camera matrix
#     """
#     _calculate_camera_intrinsics_from_camera(camera)

#     camera_matrix = np.array([
#         [camera.fx, 0,         camera.cx],
#         [0,         camera.fy, camera.cy],
#         [0,         0,         1],
#     ], dtype=np.float32)

#     return camera_matrix

# def _calculate_camera_intrinsics_from_camera(camera):
#     fx, fy, cx, cy = calculate_camera_intrinsics(
#         camera.H,
#         camera.W,
#         camera.HFOV,
#         camera.VFOV,
#     )

#     camera.fx = fx
#     camera.fy = fy
#     camera.cx = cx
#     camera.cy = cy