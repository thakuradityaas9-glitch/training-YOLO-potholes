from collections import Counter
from pathlib import Path
from PIL import Image

DATASET_ROOT = Path("dataset_model_a")
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {
    0: "pothole",
    1: "longitudinal_crack",
    2: "transverse_crack",
    3: "alligator_crack",
}


def image_map(directory):
    result = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in result:
                raise ValueError(f"Duplicate image stem: {path}")
            result[path.stem] = path
    return result


def validate_label(path, counts):
    errors = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            errors.append(f"{path}:{line_number}: expected 5 values")
            continue
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"{path}:{line_number}: non-numeric value")
            continue
        if class_id not in CLASS_NAMES:
            errors.append(f"{path}:{line_number}: invalid class ID {class_id}")
        else:
            counts[class_id] += 1
        if any(value < 0 or value > 1 for value in coordinates):
            errors.append(f"{path}:{line_number}: coordinate outside [0, 1]")
    return errors


def main():
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Converted dataset not found: {DATASET_ROOT}")
    total_counts = Counter()
    total_errors = []
    total_background = 0

    for split in SPLITS:
        image_dir = DATASET_ROOT / split / "images"
        label_dir = DATASET_ROOT / split / "labels"
        images = image_map(image_dir)
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        missing_labels = sorted(set(images) - set(labels))
        missing_images = sorted(set(labels) - set(images))
        split_counts = Counter()
        split_errors = []
        for label in labels.values():
            split_errors.extend(validate_label(label, split_counts))
        background = sum(not label.read_text(encoding="utf-8").strip() for label in labels.values())
        total_background += background
        total_counts.update(split_counts)
        total_errors.extend(split_errors)
        total_errors.extend(f"{split}: missing label for {stem}" for stem in missing_labels)
        total_errors.extend(f"{split}: missing image for {stem}" for stem in missing_images)
        print(f"{split}: images={len(images)} labels={len(labels)} background_images={background} objects={sum(split_counts.values())}")
        print(f"{split}_class_counts={dict(sorted(split_counts.items()))}")

    print(f"total_background_images={total_background}")
    print(f"total_class_counts={dict(sorted(total_counts.items()))}")
    print(f"malformed_or_invalid_annotations={len(total_errors)}")
    if total_errors:
        for error in total_errors[:20]:
            print(error)
        raise SystemExit(1)
    print("validation=passed")


if __name__ == "__main__":
    main()
