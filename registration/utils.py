import json
from pathlib import Path

def store_transform(
    name,
    source,
    parameters
):
    transform = {
        "source": source,
        "parameters": parameters
    }

    file_path = Path(f"transforms/{name}.json")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(transform, f, indent=4)

    print("Saved transform to:", file_path, "\n")

def load_parameters(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        transform = json.load(f)

    return transform["parameters"]