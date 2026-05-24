from pathlib import Path
import shutil
from PIL import Image, ImageDraw
import numpy as np
import json



project_dir = Path.cwd()

train_src = project_dir / "belt_dataset_all" / "train" / "good"
normal_test_src = project_dir / "belt_dataset_all" / "test" / "good"
defect_src = project_dir / "patches_defect_test" / "good"
ann_src = project_dir / "annotations_defect_masks"

out_root = project_dir / "belt_mvtec_defect_test"
out_category = out_root / "belt"

if out_root.exists():
    shutil.rmtree(out_root)

train_dst = out_category / "train" / "good"
test_good_dst = out_category / "test" / "good"
test_defect_dst = out_category / "test" / "defect"
gt_good_dst = out_category / "ground_truth" / "good"
gt_defect_dst = out_category / "ground_truth" / "defect"

for d in [train_dst, test_good_dst, test_defect_dst, gt_good_dst, gt_defect_dst]:
    d.mkdir(parents=True, exist_ok=True)

for p in train_src.glob("*"):
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        shutil.copy2(p, train_dst / p.name)

for p in normal_test_src.glob("*"):
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        shutil.copy2(p, test_good_dst / f"normal_{p.name}")

skipped_without_mask = 0
saved_defects = 0

for p in defect_src.glob("*"):
    if p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    json_path = ann_src / f"{p.stem}.json"

    if not json_path.exists():
        print(f"Нет json-разметки для {p.name}, пропускаю")
        skipped_without_mask += 1
        continue

    defect_name = f"defect_{p.name}"
    dst_img = test_defect_dst / defect_name

    img = Image.open(p).convert("RGB")
    w, h = img.size

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    has_defect_polygon = False

    for shape in data.get("shapes", []):
        if shape.get("label") != "defect":
            continue

        points = shape.get("points", [])

        if len(points) >= 3:
            polygon = [(float(x), float(y)) for x, y in points]
            draw.polygon(polygon, outline=255, fill=255)
            has_defect_polygon = True

    if not has_defect_polygon:
        print(f"В json нет polygon с label=defect для {p.name}, пропускаю")
        skipped_without_mask += 1
        continue

    shutil.copy2(p, dst_img)

    mask_name = f"{Path(defect_name).stem}_mask.png"
    mask.save(gt_defect_dst / mask_name)

    saved_defects += 1

dummy = Image.fromarray(np.zeros((256, 256), dtype=np.uint8))
dummy.save(gt_good_dst / "dummy.png")

print("Готово")
print(f"train/good: {len(list(train_dst.glob('*')))}")
print(f"test/good: {len(list(test_good_dst.glob('*')))}")
print(f"test/defect: {len(list(test_defect_dst.glob('*')))}")
print(f"saved: {out_root}")
print(f"test/defect with masks: {saved_defects}")
print(f"skipped without masks: {skipped_without_mask}")