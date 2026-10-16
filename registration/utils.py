import json
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

def load_image_pairs(visible_path, thermal_path, num_images):
    """Load up to num_images visible-image/temperature-array pairs.

    Select visible images in filename order and pair by case-insensitive
    stem (for example, visual/frame.jpg and thermal-data/frame.npy).
    Return (src_images, tgt_images), preserving temperature values and dtype
    in the 2D thermal arrays. OpenCV loads visible color images as BGR.
    """
    if isinstance(num_images, bool) or not isinstance(num_images, int):
        raise TypeError("num_images must be an integer.")
    if num_images <= 0:
        raise ValueError("num_images must be greater than zero.")

    visible_path = Path(visible_path)
    thermal_path = Path(thermal_path)
    for directory in (visible_path, thermal_path):
        if not directory.is_dir():
            raise NotADirectoryError(f"Image directory does not exist: {directory}")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    visible_files = sorted(
        (
            path for path in visible_path.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )[:num_images]

    if not visible_files:
        raise ValueError(f"No supported images found in {visible_path}")

    thermal_files = {}
    for path in thermal_path.iterdir():
        if path.is_file() and path.suffix.lower() == ".npy":
            stem = path.stem.casefold()
            if stem in thermal_files:
                raise ValueError(f"Multiple thermal arrays have the same stem: {path.stem}")
            thermal_files[stem] = path

    src_images, tgt_images = [], []
    for src_path in visible_files:
        tgt_path = thermal_files.get(src_path.stem.casefold())
        if tgt_path is None:
            raise FileNotFoundError(f"No thermal .npy array matches {src_path.name}")

        src_image = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
        if src_image is None:
            raise ValueError(f"Could not read image: {src_path}")

        tgt_image = np.load(tgt_path, allow_pickle=False)
        if (
            tgt_image.ndim != 2
            or tgt_image.size == 0
            or not (
                np.issubdtype(tgt_image.dtype, np.integer)
                or np.issubdtype(tgt_image.dtype, np.floating)
            )
        ):
            raise ValueError(f"Expected a nonempty 2D temperature array: {tgt_path}")

        src_images.append(src_image)
        tgt_images.append(tgt_image)

    return src_images, tgt_images

def store_transform(
    name,
    source,
    metric,
    best_metric,
    parameters,
    config,
):
    """Save parameters and a resolved snapshot of the run configuration."""
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(
            config,
            resolve=True,
            throw_on_missing=True,
            enum_to_str=True,
        )

    transform = {
        "source": source,
        "metric": metric,
        "best_metric": best_metric,
        "parameters": parameters,
        "config": config,
    }

    # Resolve and serialize before opening the output so failures do not
    # truncate an existing transform file.
    contents = json.dumps(transform, indent=4)
    file_path = Path(f"transforms/{name}.json")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        f.write(contents)
        f.write("\n")

    print("Saved transform to:", file_path, "\n")
    return file_path

def load_parameters(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        transform = json.load(f)

    return transform["parameters"]
