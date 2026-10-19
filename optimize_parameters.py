from registration.optimization import create_dataset_objective_function, run_optimization
from registration.camera_intrinsics import calculate_camera_intrinsics, get_camera_matrix
from registration.utils import load_image_pairs, store_transform

from mealpy import FloatVar, GWO, WOA
import cv2
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Optimization Algorithm Models
# - Grey Wolf Optimizer (GWO)
# - Whale Optimization Algorithm (WOA)
# - Backpropagation

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    settings = cfg.optimization

    dataset_path = Path(cfg.dataset.path)

    # load source-target image pairs
    src_images, tgt_images = load_image_pairs(
        visible_path=dataset_path / "visual",
        thermal_path=dataset_path / "thermal-data",
        num_images=settings.num_images,
    )

    # resize source to target dimensions
    src_images = [
        cv2.resize(
            src,
            (tgt.shape[1], tgt.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        for src, tgt in zip(src_images, tgt_images)
    ]

    # grayscale source images
    src_images = [
        cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        if src.ndim == 3 else src
        for src in src_images
    ]

    parameter_names = [
        "angle",
        "shear_x", "shear_y",
        "scale_x", "scale_y",
        "translation_x", "translation_y",
        "k1", "k2", "k3",
        "p1", "p2",
    ]

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

    # Camera matrix
    fx, fy, cx, cy = calculate_camera_intrinsics(
        H=thermal_height,
        W=thermal_width,
        HFOV=HFOV,
        VFOV=VFOV,
    )

    camera_matrix = get_camera_matrix(fx, fy, cx, cy)

    objective = create_dataset_objective_function(
        src_images=src_images,
        tgt_images=tgt_images,
        camera_matrix=camera_matrix,
        bins=settings.bins,
        metric=settings.metric,
        use_distortion=settings.use_distortion,
    )

    parameters, fitness = run_optimization(
        objective_function=objective,
        lower_bounds=settings.lower_bounds,
        upper_bounds=settings.upper_bounds,
        parameter_names=parameter_names,
        algorithm=settings.algorithm,
        epoch=settings.epochs,
        pop_size=settings.pop_size,
        seed=settings.seed,
    )

    store_transform(
        name=f"{cfg.dataset.name}_{settings.algorithm}",
        source=settings.algorithm,
        metric=settings.metric,
        best_metric=fitness,
        parameters=parameters,
        config=cfg,
    )

    print(f"Best {settings.metric}: {fitness}")

if __name__ == "__main__":
    main()
