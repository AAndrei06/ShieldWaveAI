from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.val(data='coco128.yaml')
metrics = results.results_dict

print(metrics)

