import flyr
from flyr.records import extract_flir_app1, parse_flir_records

import numpy as np
import cv2
from pathlib import Path

import matplotlib.pyplot as plt

DATA_ROOT = Path("data")

datasets = {"citrus", "grape", "almond"} # flir one pro datasets with incorrect labels

def read_pip_rotation(flir_path):
    """Read the stored rotation in degrees from a CIWA FLIR ONE Pro JPEG.

    Empirically identified as a little-endian float32 at byte 0x14 of the
    FLIR PiP record (0x002A). Flyr does not expose this field in pip_info.
    """
    with open(flir_path, "rb") as handle:
        flir_data = extract_flir_app1(handle)
    records = parse_flir_records(flir_data)
    if 0x002A not in records:
        raise ValueError("Missing FLIR PiP metadata")
    _, _, offset, length = records[0x002A]
    flir_data.seek(offset)
    data = flir_data.read(24)
    if length < 24 or len(data) < 24:
        raise ValueError("FLIR PiP record is too short to contain rotation")
    return float(np.frombuffer(data, dtype="<f4", count=1, offset=0x14)[0])

def crop_embedded_visible(visible, pip, rotation_deg):
    """Apply PiP scale, offsets and stored rotation; preserve image size."""
    height, width = visible.shape[:2]
    matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2),
        -rotation_deg,  # OpenCV uses the opposite sign to the stored angle.
        float(pip.real_to_ir),
    )
    offsets = np.array([pip.offset_x, pip.offset_y], dtype=np.float64)
    matrix[:, 2] -= matrix[:, :2] @ offsets
    return cv2.warpAffine(visible, matrix, (width, height), flags=cv2.INTER_LINEAR)

def visualize_crop_parameters(visible, cropped_visible, mask, alpha=0.2):

    embedded_overlay = cv2.addWeighted(
        visible, 1 - alpha,
        mask, alpha,
        0
    )

    cropped_overlay = cv2.addWeighted(
        cropped_visible, 1 - alpha,
        mask, alpha,
        0
    )

    original_cropped_overlay = cv2.addWeighted(
        cropped_original, 1 - alpha,
        mask, alpha,
        0
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    axes[0].imshow(embedded_overlay)
    axes[0].set_title('Embedded Overlay')
    axes[0].axis('off')

    axes[1].imshow(cropped_overlay)
    axes[1].set_title('Cropped Overlay')
    axes[1].axis('off')

    axes[2].imshow(original_cropped_overlay)
    axes[2].set_title('Original Cropped Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def print_crop_parameters():
    for dataset in datasets:
        for split in {"train", "valid", "test"}:
            flir_dir = DATA_ROOT / dataset / split / "raw"
            mask_dir = DATA_ROOT / dataset / split / "label"

            if not flir_dir.is_dir():
                print(f"Missing directory: {flir_dir}")
                continue

            for flir_path in sorted(flir_dir.iterdir()):
                if not flir_path.is_file():
                    continue
                if flir_path.suffix.lower() not in {".jpg", ".jpeg"}:
                    continue

                mask_path = mask_dir / f"{flir_path.stem}_L.png"
                if not mask_path.is_file():
                    print(f"Missing mask: {mask_path}")
                    continue

                thermogram = flyr.unpack(str(flir_path))
                pip = thermogram.pip_info
                if pip is None:
                    print(f"Missing PiP metadata: {flir_path}")
                    continue

                rotation_deg = read_pip_rotation(flir_path)

                print(
                    f"{flir_path}: Scale={pip.real_to_ir} "
                    f"x={pip.offset_x} y={pip.offset_y} "
                    f"deg={rotation_deg}"
                )

num = 767
flir_path = f"data/citrus/train/raw/IMG_0{num}.jpg"
mask_path = f"data/citrus/train/label/IMG_0{num}_L.png"
crop_path = f"data/cropped-visuals/citrus/train/IMG_0{num}.JPG-Visual.jpeg"

thermogram = flyr.unpack(str(flir_path))
cm = thermogram.camera_metadata
pip = thermogram.pip_info # PictureInPicture
rotation_deg = read_pip_rotation(flir_path)

visible = thermogram.optical
thermal = thermogram.celsius
mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_RGB2BGR)
cropped_original = cv2.cvtColor(cv2.imread(crop_path), cv2.COLOR_RGB2BGR)

print_crop_parameters()

print(f"Path: {flir_path}")

# Access exif camera metadata
print()
# print("Raw EXIF Data:", cm.data)
# print("Raw GPS Data:", cm.gps_data)
print(f"Camera Model: {cm.make} {cm.model}")
print(f"Timestamp: {cm.date_time}")
print(f"GPS Coordinates: {cm.gps_latitude}, {cm.gps_longitude}")

# Access picture in picture metadata
print()
print("Picture in Picture metadata:")
print(f"  Scale: {pip.real_to_ir}")
print(f"  X Offset: {pip.offset_x}")
print(f"  Y Offset: {pip.offset_y}")
print(f"  Stored Rotation: {rotation_deg} degrees")

cropped_visible = crop_embedded_visible(visible, pip, rotation_deg)
visualize_crop_parameters(visible, cropped_visible, mask)