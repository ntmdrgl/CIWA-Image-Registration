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
from matplotlib.widgets import Button

def show_registration_pair(fig):
    """Wait for Next; return False when stopped or the window is closed."""
    advance = False

    def next_pair(event):
        nonlocal advance
        advance = True
        plt.close(fig)

    def stop(event):
        plt.close(fig)

    def on_key(event):
        if event.key in ("right", "enter", " "):
            next_pair(event)
        elif event.key in ("q", "escape"):
            stop(event)

    next_button = Button(fig.add_axes([0.51, 0.02, 0.09, 0.05]), "Next")
    stop_button = Button(fig.add_axes([0.40, 0.02, 0.09, 0.05]), "Stop")
    next_button.on_clicked(next_pair)
    stop_button.on_clicked(stop)
    fig.canvas.mpl_connect("key_press_event", on_key)
    # Keep the buttons alive while the GUI event loop is running.
    plt.show(block=True)
    return advance

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    thermal_height = cfg.camera.thermal_height
    thermal_width = cfg.camera.thermal_width
    HFOV = cfg.camera.HFOV
    VFOV = cfg.camera.VFOV

    # flip dimensions if dataset is portait mode
    if cfg.dataset.is_vertical:
        thermal_height, thermal_width = thermal_width, thermal_height
        HFOV, VFOV = VFOV, HFOV

    if cfg.camera.name == "flir_one_pro":
        checkerboard = cv2.imread("checkerboard_large.png")
    elif cfg.camera.name == "flir_ax8":
        checkerboard = cv2.imread("checkerboard_small.png")

    if cfg.dataset.is_vertical:
        checkerboard = cv2.rotate(checkerboard, cv2.ROTATE_90_CLOCKWISE)

    # Load parameters from file
    parameters = load_parameters(f"transforms/{cfg.transform}.json")

    # Match optimization: resize first, with intrinsics in thermal-sized pixels.
    fx, fy, cx, cy = calculate_camera_intrinsics(
        H=thermal_height,
        W=thermal_width,
        HFOV=HFOV,
        VFOV=VFOV,
    )

    camera_matrix = get_camera_matrix(fx, fy, cx, cy)

    # Affine Matrix
    affine_matrix = get_affine_matrix(
        angle=parameters['angle'],
        hx=parameters['shear_x'],
        hy=parameters['shear_y'],
        sx=parameters['scale_x'],
        sy=parameters['scale_y'],
        tx=parameters['translation_x'],
        ty=parameters['translation_y'],
        cx=cx,
        cy=cy,
    )

    # Distortion Coefficients
    dist_coeffs = np.array([
        parameters['k1'],
        parameters['k2'],
        parameters['p1'],
        parameters['p2'],
        parameters['k3'],
    ], dtype=np.float64)

    print("Camera Matrix:", "\n", camera_matrix, "\n")
    print("Affine Matrix:", "\n", affine_matrix, "\n")
    print("Distortion Coefficients:", "\n", dist_coeffs, "\n")

    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, camera_matrix,
        (thermal_width, thermal_height), cv2.CV_32FC1,
    )
    checkerboard = cv2.resize(
        checkerboard, (thermal_width, thermal_height), interpolation=cv2.INTER_AREA,
    )
    undistorted_checkerboard = cv2.remap(
        checkerboard, map_x, map_y, interpolation=cv2.INTER_LINEAR,
    )
    transformed_checkerboard = cv2.warpAffine(
        undistorted_checkerboard, affine_matrix[:2, :], (thermal_width, thermal_height),
    )

    dataset_path = Path(cfg.dataset.path)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    visible_paths = sorted(
        (
            path for path in (dataset_path / "visual").iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not visible_paths:
        raise ValueError(f"No visible images found in {dataset_path / 'visual'}")

    thermal_paths = {}
    for path in (dataset_path / "thermal").iterdir():
        if path.is_file() and path.suffix.lower() in extensions:
            stem = path.stem.casefold()
            if stem in thermal_paths:
                raise ValueError(f"Multiple thermal images have the same stem: {path.stem}")
            thermal_paths[stem] = path

    for index, visible_path in enumerate(visible_paths, start=1):
        thermal_path = thermal_paths.get(visible_path.stem.casefold())
        if thermal_path is None:
            raise FileNotFoundError(f"No thermal image matches {visible_path.name}")

        visible_image = cv2.imread(str(visible_path))
        thermal_image = cv2.imread(str(thermal_path))
        if visible_image is None or thermal_image is None:
            raise ValueError(f"Could not read image pair: {visible_path}, {thermal_path}")

        # Apply the saved parameters in the same coordinates as optimization.
        visible_downscale = cv2.resize(
            visible_image, (thermal_width, thermal_height), interpolation=cv2.INTER_AREA,
        )
        undistorted_image = cv2.remap(
            visible_downscale, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        )
        aligned_downscale = cv2.warpAffine(
            undistorted_image, affine_matrix[:2, :], (thermal_width, thermal_height),
        )

        # Make overlays for visualization
        alpha = 0.6

        unaligned_overlay = cv2.addWeighted(
            visible_downscale, 1 - alpha,
            thermal_image, alpha,
            0
        )

        aligned_overlay = cv2.addWeighted(
            aligned_downscale, 1 - alpha,
            thermal_image, alpha,
            0
        )

        # checkerboard - transform - overlay - target
        fig, axes = plt.subplots(1, 5, figsize=(20, 6.5))
        fig.suptitle(
            f"{index}/{len(visible_paths)}: {visible_path.name}\n"
            "Next: Right/Enter/Space | Stop: Q/Esc"
        )

        axes[0].imshow(cv2.cvtColor(unaligned_overlay, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Unaligned Overlay')
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(aligned_downscale, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Aligned Visible')
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
            f"angle={parameters['angle']}°    "
            f"translation_x={parameters['translation_x']}    "
            f"translation_y={parameters['translation_y']}\n"

            f"scale_x={parameters['scale_x']}    "
            f"scale_y={parameters['scale_y']}    "
            f"shear_x={parameters['shear_x']}    "
            f"shear_y={parameters['shear_y']}\n"

            f"k1={parameters['k1']}    "
            f"k2={parameters['k2']}    "
            f"k3={parameters['k3']}    "
            f"p1={parameters['p1']}    "
            f"p2={parameters['p2']}"
        )

        fig.text(
            0.5,
            0.10,
            parameter_text,
            ha="center",
            va="bottom",
            fontsize=9,
            family="monospace",
        )

        fig.tight_layout(rect=(0, 0.23, 1, 0.90))
        if not show_registration_pair(fig):
            print(f"Stopped after viewing {index} of {len(visible_paths)} pairs.")
            break

if __name__ == "__main__":
    main()
