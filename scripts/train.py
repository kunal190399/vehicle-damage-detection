import argparse
from ultralytics import YOLO
p=argparse.ArgumentParser()
p.add_argument("--data",default="configs/dataset.yaml")
p.add_argument("--model",default="yolov8n.pt")
p.add_argument("--epochs",type=int,default=100)
p.add_argument("--imgsz",type=int,default=640)
p.add_argument("--batch",type=int,default=16)
args=p.parse_args()
YOLO(args.model).train(data=args.data,epochs=args.epochs,imgsz=args.imgsz,
                       batch=args.batch,project="runs/detect",name="damage",patience=20)
