from pathlib import Path

dataset = Path("dataset")

for split in ["train", "valid", "test"]:
    image_dir = dataset / split / "images"
    label_dir = dataset / split / "labels"

    images = list(image_dir.glob("*"))
    labels = list(label_dir.glob("*.txt"))

    image_names = {img.stem for img in images}
    label_names = {label.stem for label in labels}

    missing_labels = image_names - label_names
    missing_images = label_names - image_names

    print(f"\n{split.upper()}")
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")

    if missing_labels:
        print(f"Images without labels: {len(missing_labels)}")
    else:
        print("Every image has a label.")

    if missing_images:
        print(f"Labels without images: {len(missing_images)}")
    else:
        print("Every label has an image.")

print("\nDataset check complete.")