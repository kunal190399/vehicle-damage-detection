from pathlib import Path

def predict(model, image_path, confidence=0.25):
    results = model.predict(source=str(image_path), conf=confidence, verbose=False)
    rows = []
    for result in results:
        for box in result.boxes or []:
            class_id = int(box.cls.item())
            rows.append({
                "class_id": class_id,
                "label": str(result.names.get(class_id, class_id)),
                "confidence": round(float(box.conf.item()), 4),
                "bbox_xyxy": [round(float(x), 2) for x in box.xyxy[0].tolist()]
            })
    return rows

def load_model(weights):
    from ultralytics import YOLO
    if not Path(weights).exists() and not str(weights).endswith(".pt"):
        raise FileNotFoundError(weights)
    return YOLO(weights)
