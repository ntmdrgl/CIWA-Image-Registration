import numpy as np
import cv2
from pathlib import Path
from flyr.picture_in_picture import parse_pip_info
from flyr.records import extract_flir_app1, parse_flir_records

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
SAVE_DIR = DATA_ROOT / "uncropped-labels"
RED_BGR = (42, 42, 165)  # The masks' background is RGB (165, 42, 42).

datasets = {"almond", "citrus", "grape"} # flir one pro datasets with incorrect labels

def read_crop_parameters(flir_path):
    """Read one FLIR image's PiP scale, offsets and stored rotation."""
    with open(flir_path, "rb") as handle:
        flir_data = extract_flir_app1(handle)
    records = parse_flir_records(flir_data)
    if 0x002A not in records:
        raise ValueError(f"Missing FLIR PiP metadata: {flir_path}")
    record = records[0x002A]
    _, _, offset, length = record
    if length < 24:
        raise ValueError(f"FLIR PiP record is too short to contain rotation: {flir_path}")

    pip = parse_pip_info(flir_data, record)
    # Same empirically identified rotation field as verify_ciwa_crop_parameters.
    flir_data.seek(offset + 0x14)
    rotation_data = flir_data.read(4)
    if len(rotation_data) != 4:
        raise ValueError(f"Truncated FLIR PiP rotation: {flir_path}")
    rotation_deg = float(np.frombuffer(rotation_data, dtype="<f4", count=1)[0])
    return {
        "real_to_ir": float(pip.real_to_ir),
        "offset_x": pip.offset_x,
        "offset_y": pip.offset_y,
        "rotation_deg": rotation_deg,
    }


def uncrop_mask(mask, real_to_ir, offset_x, offset_y, rotation_deg):
    """Map a cropped BGR mask back to the embedded-visible image coordinates."""
    height, width = mask.shape[:2]
    crop_matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2), -rotation_deg, float(real_to_ir)
    )
    offsets = np.array([offset_x, offset_y], dtype=np.float64)
    crop_matrix[:, 2] -= crop_matrix[:, :2] @ offsets

    return cv2.warpAffine(
        mask,
        cv2.invertAffineTransform(crop_matrix),
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=RED_BGR,
    )


def main():
    total = 0
    for dataset in sorted(datasets):
        for split in ("train", "valid", "test"):
            mask_dir = DATA_ROOT / dataset / split / "label"
            flir_dir = DATA_ROOT / dataset / split / "raw"
            if not flir_dir.is_dir():
                print(f"Missing directory: {flir_dir}", flush=True)
                continue
            flir_images = {
                path.stem.casefold(): path
                for path in sorted(flir_dir.iterdir())
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
            }

            output_dir = SAVE_DIR / dataset / split
            output_dir.mkdir(parents=True, exist_ok=True)
            count = 0
            for mask_path in sorted(mask_dir.glob("*.png")):
                stem = mask_path.stem.removesuffix("_L")
                flir_path = flir_images.get(stem.casefold())
                if flir_path is None:
                    print(f"Missing FLIR image for mask: {mask_path}", flush=True)
                    continue
                try:
                    parameters = read_crop_parameters(flir_path)
                except ValueError as exc:
                    print(f"Skipped {mask_path.name}: {exc}", flush=True)
                    continue
                mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
                if mask is None:
                    raise ValueError(f"Could not read mask: {mask_path}")
                uncropped = uncrop_mask(mask, **parameters)
                output_path = output_dir / mask_path.name
                if not cv2.imwrite(str(output_path), uncropped):
                    raise OSError(f"Could not save mask: {output_path}")
                count += 1
            total += count
            print(f"{dataset}/{split}: saved {count} masks to {output_dir}", flush=True)
    print(f"Saved {total} uncropped masks.", flush=True)


if __name__ == "__main__":
    main()
