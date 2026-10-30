# Vehicle Damage Detection AI

YOLO-based vehicle damage detection research prototype. Includes dataset checks,
training, evaluation, CLI inference, Streamlit upload demo and unit tests.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Prepare an Ultralytics-format dataset:
```text
data/vehicle-damage/
  images/train/  images/val/  images/test/
  labels/train/  labels/val/  labels/test/
```
Each label row: `class_id x_center y_center width height` (normalised 0–1).
Edit `configs/dataset.yaml`, then:
```bash
python scripts/check_dataset.py
python scripts/train.py --epochs 100 --model yolov8n.pt
python scripts/evaluate.py --weights runs/detect/damage/weights/best.pt
python scripts/predict.py --weights runs/detect/damage/weights/best.pt --source path/to/car.jpg
streamlit run app/demo.py
pytest
```

## Dataset and annotation guidance
Use only licensed, approved, de-identified vehicle images. Define a taxonomy with
repair experts (scratch, dent, crack, broken_glass, missing_part). Split by repair
job/vehicle rather than random image to prevent near-duplicate leakage. Double-label
a subset, adjudicate disagreements, and preserve a site/time-based held-out test set.
Do not commit private imagery or trained weights.

## Evaluation and research
Report precision, recall, mAP50, mAP50-95, per-class recall, false-negative rate,
latency and performance by site/device/lighting. Compare pretrained YOLO sizes,
augmentations, confidence thresholds and (where masks exist) segmentation models.

## Limitations
A bounding box is not a repair decision. Predictions cannot establish hidden damage,
structural safety, repairability or cost. Expert review and domain validation are
mandatory. No pretrained vehicle-damage weights or ABL data are included.
