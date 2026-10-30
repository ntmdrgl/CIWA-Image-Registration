import flyr

import numpy as np
import cv2
from pathlib import Path

DATA_ROOT = Path("data")

datasets = {"almond", "citrus", "grape"} # flir one pro datasets with incorrect labels

for dataset in datasets:
    for split in {"train", "valid", "test"}:
        flir_path = DATA_ROOT / dataset / split / "raw"
        print(flir_path)

flir_path = "data/citrus/train/raw/IMG_0762.jpg"
thermogram = flyr.unpack(str(flir_path))
cm = thermogram.camera_metadata

# Access metadata dictionary and handy properties
print()
print("Raw EXIF Data:", cm.data)
print("Raw GPS Data:", cm.gps_data)

# Specific handy attributes
print()
print(f"Camera Model: {cm.make} {cm.model}")
print(f"Timestamp: {cm.date_time}")
print(f"GPS Coordinates: {cm.gps_latitude}, {cm.gps_longitude}")