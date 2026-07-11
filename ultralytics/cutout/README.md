# SAM Ship Cutout Extractor

This folder contains a batch tool for building transparent ship PNG assets from
a YOLO detection training dataset. The tool uses Meta Segment Anything Model
(SAM, not SAM2) with each YOLO bbox as a box prompt, then saves one RGBA cutout
PNG per detected ship.

The generated cutouts are intended for copy-paste augmentation workflows.

## What It Does

- Reads YOLO labels in `class_id x_center y_center width height` format.
- Converts normalized YOLO bboxes into pixel `xyxy` boxes.
- Expands each bbox slightly with `--bbox-margin` for better SAM context.
- Uses SAM `SamPredictor` with a box prompt to segment the ship.
- Saves transparent-background RGBA PNGs by class.
- Writes `metadata.csv` with source image, bbox, crop, mask area, and SAM score.
- Optionally writes `_preview` images for manual quality checks.
- Processes only the train split to avoid validation/test data leakage.

Target classes are:

- `naval` class id `0`
- `merchant` class id `1`
- `other_vessel` class id `3`

`dock` is intentionally skipped.

## Files

```text
tools/extract_ship_cutouts_sam.py
requirements_sam_cutout.txt
docs/extract_ship_cutouts_sam.md
```

The longer documentation is in `docs/extract_ship_cutouts_sam.md`.

## Install

Install dependencies:

```bash
python -m pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

If your shell does not provide `python`, use `python3`.

## SAM Checkpoint

Download a SAM checkpoint from Meta's official Segment Anything release page.
Match the checkpoint with `--sam-model-type`:

- `sam_vit_h_4b8939.pth` with `vit_h`
- `sam_vit_l_0b3195.pth` with `vit_l`
- `sam_vit_b_01ec64.pth` with `vit_b`

Example local location:

```text
checkpoints/sam_vit_h_4b8939.pth
```

## Example Usage

From this directory:

```bash
python3 img_cutouts.py \
  --images-dir ../ship/data_origin/train/images \
  --labels-dir ../ship/data_origin/train/labels_origin \
  --data-yaml ../ship/data_origin/train/classes.txt \
  --sam-checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --sam-model-type vit_h \
  --output-dir cutouts_ship \
  --target-classes naval merchant other_vessel \
  --min-bbox-area 100 \
  --bbox-margin 0.05 \
  --device cuda \
  --save-preview
```

Use `python3` instead of `python` if needed.

`--data-yaml` may point to a YOLO `data.yaml`. For the current dataset,
`../train/classes.txt` is available and can be used as a fallback class list.

## Output

```text
cutouts_sam/
  metadata.csv
  naval/
    naval_tr_0001_obj000.png
  merchant/
    merchant_tr_0002_obj001.png
  other_vessel/
    other_vessel_tr_0003_obj000.png
  _preview/
    naval_tr_0001_obj000.png
```

Each PNG is RGBA:

- alpha `255` inside the SAM ship mask
- alpha `0` outside the mask
- crop bounds based on the mask bbox

## Preview Check

Run with `--save-preview` and inspect `cutouts_sam/_preview`.

Preview colors:

- yellow: original YOLO bbox
- blue: expanded SAM prompt bbox
- green: final crop bbox
- red overlay: selected SAM mask

Use previews to remove bad cutouts or tune `--bbox-margin`,
`--min-bbox-area`, and `--min-mask-area`.

## Data Leakage Warning

Do not generate cutouts from `val` or `test`. The script rejects paths that
contain `val`, `validation`, or `test`, and requires both image and label paths
to include `train`.

Only train-derived cutouts should be used by the copy-paste augmentation GUI.
