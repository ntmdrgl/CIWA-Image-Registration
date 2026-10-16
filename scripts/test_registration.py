from registration.camera_intrinsics import calculate_camera_intrinsics, get_camera_matrix
from registration.transformations import get_affine_matrix
from registration.utils import load_parameters

import numpy as np
import cv2
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import matplotlib.pyplot as plt

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    visible_height = cfg.camera.image_height
    visible_width = cfg.camera.image_width
    thermal_height = cfg.camera.thermal_height
    thermal_width = cfg.camera.thermal_width
    HFOV = cfg.camera.HFOV
    VFOV = cfg.camera.VFOV

    # flip dimensions if dataset is portait mode
    if cfg.dataset.is_vertical:
        visible_height, visible_width = visible_width, visible_height
        thermal_height, thermal_width = thermal_width, thermal_height
        HFOV, VFOV = VFOV, HFOV

    if cfg.camera.name == "flir_one_pro":
        checkerboard = cv2.imread("checkerboard_large.png")
    elif cfg.camera.name == "flir_ax8":
        checkerboard = cv2.imread("checkerboard_small.png")

    if cfg.dataset.is_vertical:
        checkerboard = cv2.rotate(checkerboard, cv2.ROTATE_90_CLOCKWISE)

    # Load parameters from file
    parameters = load_parameters("transforms/manual_transform.json")

    # Camera matrix
    fx, fy, cx, cy = calculate_camera_intrinsics(
        H=visible_height,
        W=visible_width,
        HFOV=HFOV,
        VFOV=VFOV,
    )
    
    camera_matrix = get_camera_matrix(fx, fy, cx, cy)
    
    # Affine Matrix
    affine_matrix = get_affine_matrix(
        angle=parameters["angle"], 
        hx=parameters["shear_x"], 
        hy=parameters["shear_y"], 
        sx=parameters["scale_x"], 
        sy=parameters["scale_y"], 
        tx=parameters["translation_x"], 
        ty=parameters["translation_y"], 
        cx=cx, 
        cy=cy,
    )

    # Distortion Coefficients
    dist_coeffs = np.array([
        parameters["k1"], 
        parameters["k2"],
        parameters["p1"],
        parameters["p2"],
        parameters["k3"],
    ], dtype=np.float32)

    print("Camera Matrix:", "\n", camera_matrix, "\n")
    print("Affine Matrix:", "\n", affine_matrix, "\n")
    print("Distortion Coefficients:", "\n", dist_coeffs, "\n")
    
    visible_path = next((Path(cfg.dataset.path) / "visual").iterdir())
    visible_image = cv2.imread(str(visible_path))

    thermal_path = next((Path(cfg.dataset.path) / "thermal").iterdir())
    thermal_image = cv2.imread(str(thermal_path))

    # Apply transformations to visible image
    undistorted_image = cv2.undistort(visible_image, camera_matrix, dist_coeffs)
    transformed_image = cv2.warpAffine(undistorted_image, affine_matrix[:2, :], (visible_width, visible_height))

    undistorted_checkerboard = cv2.undistort(checkerboard, camera_matrix, dist_coeffs)
    transformed_checkerboard = cv2.warpAffine(undistorted_checkerboard, affine_matrix[:2, :], (visible_width, visible_height))

    # Make overlays for visualization
    alpha = 0.5

    visible_downscale = cv2.resize(visible_image, (thermal_width, thermal_height), interpolation=cv2.INTER_AREA)
    unaligned_overlay = cv2.addWeighted(
        visible_downscale, 1 - alpha,
        thermal_image, alpha,
        0
    )

    aligned_downscale = cv2.resize(transformed_image, (thermal_width, thermal_height), interpolation=cv2.INTER_AREA)
    aligned_overlay = cv2.addWeighted(
        aligned_downscale, 1 - alpha,
        thermal_image, alpha,
        0
    )

    # checkerboard - transform - overlay - target
    fig, axes = plt.subplots(1, 5, figsize=(20, 6.5))

    axes[0].imshow(cv2.cvtColor(visible_downscale, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Unaligned Visible')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(unaligned_overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Unaligned Overlay')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(aligned_overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Aligned Overlay')
    axes[2].axis('off')

    axes[3].imshow(cv2.cvtColor(thermal_image, cv2.COLOR_BGR2RGB))
    axes[3].set_title('Thermal Target')
    axes[3].axis('off')
    
    axes[4].imshow(cv2.cvtColor(transformed_checkerboard, cv2.COLOR_BGR2RGB))
    axes[4].set_title('Checkerboard')
    axes[4].axis('off') 

    parameter_text = (
        f"angle={parameters["angle"]}°    "
        f"translation_x={parameters["translation_x"]}    "
        f"translation_y={parameters["translation_y"]}\n"

        f"scale_x={parameters["scale_x"]}    "
        f"scale_y={parameters["scale_y"]}    "
        f"shear_x={parameters["shear_x"]}    "
        f"shear_y={parameters["shear_y"]}\n"

        f"k1={parameters["k1"]}    "
        f"k2={parameters["k2"]}    "
        f"k3={parameters["k3"]}    "
        f"p1={parameters["p1"]}    "
        f"p2={parameters["p2"]}"
    )

    fig.text(
        0.5,
        0.02,
        parameter_text,
        ha="center",
        va="bottom",
        fontsize=9,
        family="monospace",
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()