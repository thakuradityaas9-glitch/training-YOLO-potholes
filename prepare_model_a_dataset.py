from pathlib import Path
import shutil

SOURCE_ROOT = Path("dataset")
OUTPUT_ROOT = Path("dataset_model_a")
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Source IDs approved for Model A and their new four-class IDs.
CLASS_MAP = {0: 1, 1: 1, 3: 2, 4: 2, 5: 3, 6: 0}
SOURCE_NAMES = {
    0: "D00",
    1: "D01",
    2: "D0w0",
    3: "D10",
    4: "D11",
    5: "D20",
    6: "D40",
    7: "D43",
    8: "D44",
    9: "D50",
}
CLASS_NAMES = {
    0: "pothole",
    1: "longitudinal_crack",
    2: "transverse_crack",
    3: "alligator_crack",
}
EXCLUDED_IDS = set(SOURCE_NAMES) - set(CLASS_MAP)


def image_stem_map(image_dir):
    images = {}
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in images:
                raise RuntimeError(f"Duplicate image stem in {image_dir}: {path.stem}")
            images[path.stem] = path
    return images


def parse_label(label_path):
    converted = []
    source_ids = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 values, got {len(fields)}")
        try:
            source_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(f"{label_path}:{line_number}: non-numeric annotation") from error
        if source_id not in SOURCE_NAMES:
            raise ValueError(f"{label_path}:{line_number}: unknown source class {source_id}")
        if any(value < 0 or value > 1 for value in coordinates):
            raise ValueError(f"{label_path}:{line_number}: coordinates must be normalized to [0, 1]")
        source_ids.append(source_id)
        if source_id in CLASS_MAP:
            converted.append(f"{CLASS_MAP[source_id]} {' '.join(fields[1:])}")
    return converted, source_ids


def main():
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE_ROOT}")
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT_ROOT}")

    summary = {split: {"images": 0, "labels": 0, "background": 0, "retained_objects": 0, "excluded_objects": 0} for split in SPLITS}
    class_counts = {class_id: 0 for class_id in CLASS_NAMES}
    OUTPUT_ROOT.mkdir()

    try:
        for split in SPLITS:
            source_images = image_stem_map(SOURCE_ROOT / split / "images")
            source_labels = {path.stem: path for path in (SOURCE_ROOT / split / "labels").glob("*.txt")}
            if set(source_images) != set(source_labels):
                missing_labels = sorted(set(source_images) - set(source_labels))
                missing_images = sorted(set(source_labels) - set(source_images))
                raise RuntimeError(f"{split}: missing_labels={missing_labels[:5]}, missing_images={missing_images[:5]}")

            output_images = OUTPUT_ROOT / split / "images"
            output_labels = OUTPUT_ROOT / split / "labels"
            output_images.mkdir(parents=True)
            output_labels.mkdir(parents=True)

            for stem, source_image in sorted(source_images.items()):
                converted, source_ids = parse_label(source_labels[stem])
                shutil.copy2(source_image, output_images / source_image.name)
                (output_labels / f"{stem}.txt").write_text(
                    "\n".join(converted) + ("\n" if converted else ""), encoding="utf-8"
                )
                stats = summary[split]
                stats["images"] += 1
                stats["labels"] += 1
                stats["retained_objects"] += len(converted)
                stats["excluded_objects"] += sum(source_id in EXCLUDED_IDS for source_id in source_ids)
                if not converted:
                    stats["background"] += 1
                for line in converted:
                    class_counts[int(line.split()[0])] += 1
    except Exception:
        shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
        raise

    print("Conversion complete")
    print(f"source={SOURCE_ROOT}")
    print(f"output={OUTPUT_ROOT}")
    for split, stats in summary.items():
        print(f"{split}: {stats}")
    print("objects_by_model_a_class=")
    for class_id, name in CLASS_NAMES.items():
        print(f"  {class_id} {name}: {class_counts[class_id]}")
    print(f"excluded_source_ids={sorted(EXCLUDED_IDS)}")
    print("background_images_preserved=true")


if __name__ == "__main__":
    main()
