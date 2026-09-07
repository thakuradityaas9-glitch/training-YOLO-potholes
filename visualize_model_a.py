from pathlib import Path
import random
from PIL import Image, ImageDraw

DATASET_ROOT = Path("dataset_model_a")
OUTPUT_ROOT = Path("visualized_model_a")
CLASS_NAMES = {
    0: "pothole",
    1: "longitudinal_crack",
    2: "transverse_crack",
    3: "alligator_crack",
}
COLORS = {0: "red", 1: "blue", 2: "yellow", 3: "lime"}


def draw_sample(image_path, label_path, output_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        class_id = int(fields[0])
        center_x, center_y, box_width, box_height = map(float, fields[1:])
        left = (center_x - box_width / 2) * width
        top = (center_y - box_height / 2) * height
        right = (center_x + box_width / 2) * width
        bottom = (center_y + box_height / 2) * height
        color = COLORS[class_id]
        draw.rectangle((left, top, right, bottom), outline=color, width=max(2, width // 400))
        draw.text((left + 3, top + 3), CLASS_NAMES[class_id], fill=color)
    image.save(output_path)


def main():
    random.seed(20260907)
    image_dir = DATASET_ROOT / "train" / "images"
    label_dir = DATASET_ROOT / "train" / "labels"
    images = sorted(path for path in image_dir.iterdir() if path.is_file())
    samples = random.sample(images, min(10, len(images)))
    OUTPUT_ROOT.mkdir(exist_ok=True)
    for old_sample in OUTPUT_ROOT.glob("*.jpg"):
        old_sample.unlink()
    for image_path in samples:
        output_path = OUTPUT_ROOT / image_path.name
        draw_sample(image_path, label_dir / f"{image_path.stem}.txt", output_path)
        print(output_path)
    print(f"generated_samples={len(samples)}")


if __name__ == "__main__":
    main()
