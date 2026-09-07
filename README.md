# Model A — Pavement Damage Detection

YOLO-based object detection model for detecting **pavement and road-surface damage** from road images and video frames.

Model A is the pavement-damage component of the larger road-condition monitoring system.

---

## 🎯 Objective

Model A focuses specifically on **damage to the road surface and road edges**.

The model is trained using the **RDD2022-India** dataset, which contains road-damage annotations from Indian road scenes.

Dataset Version 5 contains **7,706 images**:

```text
Train:      6,532 images
Validation:   786 images
Test:         388 images
```

The dataset is licensed under **CC BY 4.0**.

---

# 📊 Dataset

### RDD2022-India — Version 5

[Download / Dataset — Roboflow](https://universe.roboflow.com/prakhar-kpb1v/rdd2022-india-il8ju/dataset/5)

The dataset provides YOLO11-compatible annotations and configuration files.

**License:** CC BY 4.0

**Total images:** 7,706

**Dataset split:**

| Split      | Images |
| ---------- | -----: |
| Train      |  6,532 |
| Validation |    786 |
| Test       |    388 |

No preprocessing or augmentations were applied to this version of the dataset on Roboflow.

---

# 🏷️ Original Dataset Classes

RDD2022-India contains 10 road-damage classes:

```text
D00
D01
D0w0
D10
D11
D20
D40
D43
D44
D50
```

These are the original RDD2022 road-damage labels.

For reference, the project uses the following semantic interpretation of these classes:

| Original ID | Damage Type                                   |
| ----------- | --------------------------------------------- |
| `D00`       | Longitudinal crack                            |
| `D01`       | Longitudinal crack with other characteristics |
| `D0w0`      | Longitudinal crack / related road marking     |
| `D10`       | Transverse crack                              |
| `D11`       | Transverse crack with other characteristics   |
| `D20`       | Alligator crack                               |
| `D40`       | Pothole                                       |
| `D43`       | Rutting / road deformation                    |
| `D44`       | Road-edge related damage                      |
| `D50`       | Other road-surface damage                     |

> The exact class semantics should always be checked against the original RDD2022 annotation documentation when interpreting individual labels.

---

# 🧠 Model A Target Classes

For the larger road-condition system, Model A is intended to produce a more application-oriented set of pavement-damage categories.

The planned categories are:

```text
0 pothole
1 longitudinal_crack
2 transverse_crack
3 alligator_crack
4 damaged_road_edge
```

These categories are intended to make the output easier to consume by the downstream complaint/reporting system.

---

# 🔄 Model Architecture

The overall project separates road problems into specialized models.

```text
                    Road Image / Video
                           │
                           ▼
              ┌─────────────────────────┐
              │       Model A            │
              │   Pavement Damage        │
              │                         │
              │ • Potholes               │
              │ • Longitudinal cracks    │
              │ • Transverse cracks      │
              │ • Alligator cracks       │
              │ • Damaged road edges     │
              └─────────────────────────┘
                           │
                           │
              ┌─────────────────────────┐
              │       Model B            │
              │    Road Hazards          │
              │                         │
              │ • Debris                 │
              │ • Waterlogging           │
              │ • Open manholes          │
              │ • Fallen objects         │
              │ • Construction           │
              └─────────────────────────┘
                           │
                           │
              ┌─────────────────────────┐
              │       Model C            │
              │ Infrastructure / Signs   │
              └─────────────────────────┘
                           │
                           ▼
                 Detected Road Problems
```

Separating the models allows each model to specialize in a smaller problem domain rather than forcing one model to detect every possible road condition.

---

# 📁 Project Structure

The project should look approximately like:

```text
road-damage-ai/
│
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   │
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   │
│   └── test/
│       ├── images/
│       └── labels/
│
├── train.py
├── detect.py
├── check_dataset.py
├── visualize_dataset.py
├── data.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

Large files such as:

```text
dataset/
runs/
*.pt
venv/
```

should not be committed to GitHub.

---

# 💻 Training on the Gaming PC

## 1. Clone the repository

```powershell
git clone https://github.com/thakuradityaas9-glitch/training-YOLO-potholes.git
cd training-YOLO-potholes
```

> If the Model A repository name has since been changed, use the current GitHub repository URL instead.

---

# 2. Create a virtual environment

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

You should see:

```text
(venv)
```

at the beginning of the PowerShell prompt.

---

# 3. Install dependencies

```powershell
pip install -r requirements.txt
```

Then verify the Ultralytics installation:

```powershell
yolo checks
```

---

# 📥 4. Download RDD2022-India

Open:

[RDD2022-India — Roboflow Version 5](https://universe.roboflow.com/prakhar-kpb1v/rdd2022-india-il8ju/dataset/5)

Download the dataset using the **YOLO11** format. Roboflow provides a YOLO11 export for this version.

Extract the dataset into the project.

The final structure should contain:

```text
dataset/
├── train/
├── valid/
└── test/
```

with images and labels inside each split.

---

# 🔍 5. Verify `data.yaml`

The dataset configuration should point to:

```yaml
train: dataset/train/images
val: dataset/valid/images
test: dataset/test/images
```

and contain the correct number of classes and class names from the downloaded RDD2022-India version.

**Do not manually change the class IDs unless the training pipeline is intentionally converting them.**

---

# ✅ 6. Validate the dataset

Run:

```powershell
python check_dataset.py
```

The checker should verify:

* YOLO annotation format
* Valid class IDs
* Valid normalized bounding boxes
* Missing images
* Missing labels
* Invalid annotations
* Object counts

Do not start a long training run until the dataset passes validation.

---

# 👁️ 7. Visualize annotations

Before training, inspect some random annotated images:

```powershell
python visualize_dataset.py --count 10
```

This generates visualized images outside the original dataset.

Check that:

* Boxes are positioned correctly
* Boxes surround the actual damage
* Class names match the visible damage
* No obvious annotation errors exist

---

# 🧠 8. Train YOLO

For serious training on the gaming PC, start with **YOLO11s** rather than the tiny `YOLO11n` model.

Example:

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=-1,
    device=0,
    patience=20,
    workers=8,
    project="runs/model_a",
    name="yolo11s_pavement_damage"
)
```

Then run:

```powershell
python train.py
```

---

# ⚙️ Training Parameters

| Parameter  |     Value | Purpose                            |
| ---------- | --------: | ---------------------------------- |
| Model      | `YOLO11s` | Starting model                     |
| Epochs     |     `100` | Training duration                  |
| Image size |     `640` | Input resolution                   |
| Batch      |      `-1` | Automatically determine batch size |
| Device     |       `0` | First CUDA GPU                     |
| Patience   |      `20` | Early stopping                     |
| Workers    |       `8` | Data-loading workers               |

If the GPU runs out of VRAM, manually reduce:

```python
batch=8
```

or:

```python
batch=4
```

---

# 📈 9. Training Results

Training results will be stored inside:

```text
runs/model_a/
```

The best checkpoint should normally be:

```text
runs/model_a/yolo11s_pavement_damage/weights/best.pt
```

`best.pt` is the trained model checkpoint selected according to the validation performance during training.

---

# 🔍 10. Run Detection

Once training is complete, test the model on an image:

```powershell
yolo detect predict model=runs/model_a/yolo11s_pavement_damage/weights/best.pt source="test_image.jpg" conf=0.25
```

Or use the project's detection script:

```powershell
python detect.py
```

---

# 📊 11. Model Evaluation

When evaluating the trained model, pay attention to:

* mAP@50
* mAP@50-95
* Precision
* Recall
* Per-class AP
* Confusion matrix

Per-class performance is particularly important because some types of road damage are significantly harder to detect than others.

---

# 🚧 Current Limitations

The RDD2022-India dataset is useful for training a road-damage detector, but the final application may encounter conditions not fully represented in the dataset.

Potential limitations include:

* Different Indian road surfaces
* Night-time images
* Rain and poor visibility
* Heavy traffic
* Small or partially visible potholes
* Severe occlusion
* Low-quality CCTV footage
* Different camera heights and angles
* Damage hidden by vehicles
* Unusual road materials

Additional project-specific data should eventually be collected to improve robustness.

---

# 🚀 Future Improvements

Potential improvements include:

* Compare YOLO11s and YOLO11m
* Train at higher image resolution
* Add Indian dashcam/CCTV footage
* Add night-time examples
* Add rainy-weather examples
* Add difficult/partially occluded examples
* Perform hard-negative mining
* Tune confidence and IoU thresholds
* Evaluate per-class performance
* Add additional road-edge examples
* Export the final model for optimized inference

---

# 🔗 Dataset & Repository

### Dataset

[RDD2022-India — Roboflow v5](https://universe.roboflow.com/prakhar-kpb1v/rdd2022-india-il8ju/dataset/5)

### GitHub

[Model A — GitHub](https://github.com/thakuradityaas9-glitch/training-YOLO-potholes)

---

## 📜 Dataset License

The RDD2022-India dataset used by this project is listed as **CC BY 4.0** on Roboflow.

The dataset remains subject to its original license and attribution requirements.

The code in this repository is separate from the dataset and should not be interpreted as changing the dataset's licensing terms.
