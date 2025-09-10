# Body Measurement Prediction from Pose Estimation

This project implements a pipeline that takes **2D pose keypoints** from images (front and side views) and predicts **body circumferences** (chest, waist, hips, thighs, calves, etc.).  
It combines **YOLOv8 Pose** for keypoint extraction, custom geometric feature engineering, and a **Ridge regression model** trained with cross-validation.  
Additionally, it includes tools for **visual overlays** and **pixel-to-cm body measurements** using silhouette masks.

---

## 📂 Project Structure

```
.
├── data/
│   ├── labels_with_gender.csv   # Ground-truth body measurements + metadata
│   └── images/                  # Input images (id_front.png, id_side.png)
├── features/
│   └── yolo_features.csv        # Extracted features from keypoints
├── models/
│   └── yolo_ridge.joblib        # Trained regression model (saved artifact)
├── reports/
│   ├── cv_summary_ridgecv.csv   # Cross-validation metrics
│   ├── cv_folds_ridgecv.csv     # Per-fold results
│   └── holdout_validation.csv   # Final 80/20 split validation
├── code.ipynb                   # Main training and validation notebook
├── pre_processing.ipynb         # Pre-processing and feature engineering
└── README.md
```

---

## 🚀 Pipeline Overview

1. **Pose Estimation**  
   - Uses [Ultralytics YOLOv8-Pose](https://docs.ultralytics.com/tasks/pose/) to detect 17 COCO keypoints.
   - Supports **front** and **side** images (if available).

2. **Feature Extraction**  
   - Geometric distances between keypoints (e.g., shoulder span, hip span).  
   - Normalization by **front_height_px** to make features scale-invariant.

3. **Model Training**  
   - Baselines: `LinearRegression` and `RidgeCV`.  
   - RidgeCV chosen via 5-fold cross-validation (macro MAE ≈ 3.8 cm).  
   - Metrics: **MAE** (cm) and **RMSE** (cm), both per measurement and macro-averaged.  
   - Reports saved in `reports/`.

4. **Validation**  
   - Stratified K-Fold split by gender (if available).  
   - Final 80/20 holdout validation to simulate deployment.  
   - Example errors (MAE, cm): chest 5.23, waist 6.85, hips 3.80, calves 1.97.

5. **Inference**  
   - Load model artifact (`yolo_ridge.joblib`).  
   - Run YOLO pose on new image(s).  
   - Extract features → align with saved `feature_names`.  
   - Predict circumferences.  
   - Results returned as a Pandas DataFrame.

6. **Visualization & Measurement**  
   - Segments body parts (chest, waist, hips, arms, thighs, calves).  
   - Clips to silhouette via Otsu thresholding.  
   - Supports **red measurement lines**:  
     - Horizontal for torso/legs  
     - Vertical for arms  
   - Converts pixel widths → cm using known person height.

---

## 🛠 Requirements

- Python 3.10+
- [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv8)
- scikit-learn ≥ 1.6
- pandas, numpy, matplotlib, seaborn
- Pillow, imageio

Install with:

```bash
pip install -r requirements.txt
```

---

## ⚡ Usage

### 1. Train & Validate
Run the notebook:

```bash
jupyter notebook code.ipynb
```

This will:
- Load labels and features.
- Train RidgeCV with cross-validation.
- Save the model to `models/yolo_ridge.joblib`.
- Export metrics to `reports/`.

### 2. Inference on New Image

```python
from ultralytics import YOLO
import joblib, pandas as pd
import os
from extract_features_yolov8 import run_pose, feature_vector

# Parameters
ID = "syn_f000000-0-Pre"
IMGDIR = "data/mulheres_15k"
WEIGHTS = "yolov8m-pose.pt"
MODEL_PATH = "models/yolo_ridge.joblib"

# File paths
front = os.path.join(IMGDIR, f"{ID}/front.png")
side  = os.path.join(IMGDIR, f"{ID}/side.png")

# Run YOLO pose
model_pose = YOLO(WEIGHTS)
fkp = run_pose(model_pose, front)
skp = run_pose(model_pose, side) if os.path.exists(side) else None

# Extract features & normalize
feats, _ = feature_vector(fkp, skp)
H = max(feats["front_height_px"], 1.0)
feats_norm = {k:(v/H) if k.endswith("_px") else v for k,v in feats.items()}

# Align with model features
pack = joblib.load(MODEL_PATH)
feat_cols = pack["feature_names"]
X = pd.DataFrame([feats_norm])[feat_cols].values

# Predict
yhat = pack["model"].predict(X)[0]
print(dict(zip(pack["targets"], yhat)))
```

### 3. Visual Overlay & Measurements

```python
from measurements import build_part_masks, measure_horizontal_lines, normalize_keypoints

fkp_n = normalize_keypoints(fkp)
parts_masks, body_mask, overlay = build_part_masks(front, fkp_n)

front_height_px = feats["front_height_px"]
HEIGHT_CM = 170.0  # known person height
measurements = measure_horizontal_lines(front, fkp_n, parts_masks, body_mask,
                                        front_height_px, HEIGHT_CM, plot=True)

print(measurements)
```

---

## 📊 Results

- **Macro MAE:** ~3.8 cm across all circumferences.  
- **Best performing parts:** calf, neck, knee (MAE < 2.5 cm).  
- **Challenging parts:** waist and abdomen (MAE > 6 cm).  
- Visual overlays allow manual inspection of segmentation and measurement logic.

---

## 🔮 Next Steps

- Enrich feature set (e.g., direct measurements from red lines instead of normalized spans).  
- Explore non-linear models (e.g., Gradient Boosting, LightGBM).  
- Add unit tests with `pytest`.  
- Improve robustness against segmentation failures with keypoint-only fallback.  
- Package as CLI or Streamlit app for interactive use.

---

## 📜 License

MIT License.
