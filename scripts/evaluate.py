import argparse, json
from pathlib import Path
from ultralytics import YOLO
p=argparse.ArgumentParser(); p.add_argument("--weights",required=True)
p.add_argument("--data",default="configs/dataset.yaml"); p.add_argument("--split",default="test")
p.add_argument("--output",default="outputs/metrics.json"); args=p.parse_args()
m=YOLO(args.weights).val(data=args.data,split=args.split)
metrics={"precision":float(m.box.mp),"recall":float(m.box.mr),
         "map50":float(m.box.map50),"map50_95":float(m.box.map)}
Path(args.output).parent.mkdir(parents=True,exist_ok=True)
Path(args.output).write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
