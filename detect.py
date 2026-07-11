import warnings
warnings.filterwarnings('ignore')
import shutil
from pathlib import Path

from ultralytics import YOLO


def organize_outputs(save_dir: Path) -> None:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".mp4", ".avi", ".mov", ".mkv"}
    images_dir = save_dir / "images"
    labels_dir = save_dir / "labels"
    labels_src = save_dir / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    temp_labels_dir = save_dir / "labels_tmp"
    if labels_src.exists() and labels_src.is_dir():
        labels_src.rename(temp_labels_dir)

    labels_dir.mkdir(parents=True, exist_ok=True)

    for item in save_dir.iterdir():
        if item.is_file() and item.suffix.lower() in image_exts:
            shutil.move(str(item), str(images_dir / item.name))

    if temp_labels_dir.exists() and temp_labels_dir.is_dir():
        for txt in temp_labels_dir.iterdir():
            if txt.is_file():
                shutil.move(str(txt), str(labels_dir / txt.name))
        temp_labels_dir.rmdir()


if __name__ == '__main__':
    model = YOLO("runs_merge/11n_asff/weights/best.pt")
    project = "runs/predict"
    name = "baseline_merge"

    results = model.predict(
        source="ultralytics/ship_plane/data/test/images",
        save=True,
        save_txt=True,
        save_conf=True,
        project=project,
        name=name,
        imgsz=960,
        device='0',
        conf=0.001
        )

    if results:
        save_dir = Path(results[0].save_dir)
    else:
        save_dir = Path(project) / name

    organize_outputs(save_dir)

