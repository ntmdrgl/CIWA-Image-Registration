import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
UNCROPPED_DIR = DATA_ROOT / "uncropped-labels"
CROPPED_DIR = DATA_ROOT / "cropped-visuals"
SAVE_DIR = DATA_ROOT / "verify-uncropped-labels"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def index_visible_images(directory, cropped=False):
    """Match visible filenames to mask stems, including FLIR export suffixes."""
    images = {}
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            stem = path.stem
            if cropped:
                if stem.lower().endswith("-visual"):
                    stem = stem[:-len("-visual")]
                if Path(stem).suffix.lower() in IMAGE_EXTENSIONS:
                    stem = Path(stem).stem
            images[stem.casefold()] = path
    return images


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def visualize_crop_parameters(embedded_visible, uncropped_mask, cropped_visible,
                              cropped_visible_mask, output_path, alpha=0.2, title=""):
    """Save two overlays of BGR arrays, without altering image or mask geometry."""
    pairs = (
        (embedded_visible, uncropped_mask, "Embedded visible + uncropped mask"),
        (cropped_visible, cropped_visible_mask, "Cropped visible + original mask"),
    )
    for visible, mask, label in pairs:
        if visible.shape != mask.shape:
            raise ValueError(f"{label}: image shape {visible.shape} != mask shape {mask.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), layout="constrained")
    try:
        if title:
            fig.suptitle(title, fontsize=10)
        for axis, (visible, mask, label) in zip(axes, pairs):
            overlay = cv2.addWeighted(visible, 1 - alpha, mask, alpha, 0)
            axis.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), interpolation="nearest")
            axis.set_title(label, fontsize=10)
            axis.axis("off")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    finally:
        plt.close(fig)


def save_pair(pair):
    """Render one comparison in an isolated process; return any input error."""
    embedded_path, uncropped_path, cropped_path, original_mask_path, output_path, title = pair
    try:
        inputs = (embedded_path, uncropped_path, cropped_path, original_mask_path)
        if output_path.is_file() and output_path.stat().st_mtime >= max(
            path.stat().st_mtime for path in inputs
        ):
            return "existing", None
        visualize_crop_parameters(
            read_image(embedded_path), read_image(uncropped_path),
            read_image(cropped_path), read_image(original_mask_path),
            output_path, title=title,
        )
        return "saved", None
    except (ValueError, OSError) as exc:
        return "skipped", str(exc)


def main():
    if not UNCROPPED_DIR.is_dir():
        raise FileNotFoundError(f"Missing uncropped mask directory: {UNCROPPED_DIR}")

    jobs = []
    skipped = []
    for dataset_dir in sorted(path for path in UNCROPPED_DIR.iterdir() if path.is_dir()):
        dataset = dataset_dir.name
        for split in ("train", "valid", "test"):
            mask_dir = dataset_dir / split
            if not mask_dir.is_dir():
                continue
            source_dir = DATA_ROOT / dataset / split
            embedded_images = index_visible_images(source_dir / "visual")
            cropped_images = index_visible_images(CROPPED_DIR / dataset / split, cropped=True)
            for uncropped_path in sorted(mask_dir.glob("*.png")):
                stem = uncropped_path.stem.removesuffix("_L")
                embedded_path = embedded_images.get(stem.casefold())
                cropped_path = cropped_images.get(stem.casefold())
                original_mask_path = source_dir / "label" / uncropped_path.name
                try:
                    missing = []
                    if embedded_path is None:
                        missing.append("embedded visible")
                    if cropped_path is None:
                        missing.append("cropped visible")
                    if not original_mask_path.is_file():
                        missing.append("original mask")
                    if missing:
                        raise ValueError("Missing " + ", ".join(missing))
                    jobs.append((
                        embedded_path, uncropped_path, cropped_path, original_mask_path,
                        SAVE_DIR / dataset / split / f"{uncropped_path.stem}.png",
                        f"{dataset}/{split}/{stem}",
                    ))
                except (ValueError, OSError) as exc:
                    skipped.append({"mask": str(uncropped_path.relative_to(DATA_ROOT)),
                                    "reason": str(exc)})
                    print(f"Skipped {uncropped_path.name}: {exc}", flush=True)
                    continue

    saved = 0
    existing = 0
    workers = min(4, os.cpu_count() or 1)
    print(f"Preparing {len(jobs)} comparison figures with {workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, (pair, result) in enumerate(zip(jobs, executor.map(save_pair, jobs)), 1):
            status, error = result
            if error is not None:
                skipped.append({"mask": str(pair[1].relative_to(DATA_ROOT)), "reason": error})
                print(f"Skipped {pair[1].name}: {error}", flush=True)
            elif status == "existing":
                existing += 1
            else:
                saved += 1
            if index % 100 == 0 or index == len(jobs):
                print(f"Checked {index}/{len(jobs)} figures...", flush=True)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with (SAVE_DIR / "skipped.json").open("w", encoding="utf-8") as handle:
        json.dump(skipped, handle, indent=2)
    print(f"Saved {saved} figures to {SAVE_DIR}; reused {existing}; "
          f"skipped {len(skipped)} pairs.", flush=True)


if __name__ == "__main__":
    main()
