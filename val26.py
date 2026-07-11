from ultralytics import YOLO

model = YOLO('runs/detect/runs_ship/26n_baseline/weights/best.pt')
metrics = model.val(
    data='ultralytics/cfg/datasets/data_origin.yaml',
    split='test',
    imgsz=960,
    batch=16,
    device=0,
    save_json=True,
)
print("mAP50-95 (test):", metrics.box.map)
print("mAP50 (test):   ", metrics.box.map50)
print("mAP75 (test):   ", metrics.box.map75)
print("Per-class AP:", metrics.box.maps)
