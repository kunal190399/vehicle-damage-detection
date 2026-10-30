import argparse, json
from pathlib import Path
from ultralytics import YOLO
from damage_ai.inference import predict
p=argparse.ArgumentParser(); p.add_argument("--weights",required=True)
p.add_argument("--source",required=True); p.add_argument("--output",default="outputs")
p.add_argument("--conf",type=float,default=.25); args=p.parse_args()
model=YOLO(args.weights); src=Path(args.source)
files=[src] if src.is_file() else [x for x in src.rglob("*") if x.suffix.lower() in {".jpg",".jpeg",".png"}]
out=Path(args.output); out.mkdir(parents=True,exist_ok=True); results=[]
for f in files:
    results.append({"image":str(f),"detections":predict(model,f,args.conf)})
    model.predict(str(f),conf=args.conf,save=True,project=str(out),name="annotated",exist_ok=True)
(out/"predictions.json").write_text(json.dumps(results,indent=2))
print(f"Processed {len(files)} images.")
