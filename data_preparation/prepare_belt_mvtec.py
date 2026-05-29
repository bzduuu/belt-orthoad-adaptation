from pathlib import Path
import shutil
from PIL import Image
import numpy as np


project_dir = Path.cwd()

src_dataset = project_dir / "belt_dataset_all"
out_root = project_dir / "belt_mvtec_all"
out_category = out_root / "belt"

if out_root.exists():
    shutil.rmtree(out_root)

train_src = src_dataset / "train" / "good"
test_src = src_dataset / "test" / "good"

train_dst = out_category / "train" / "good"
test_dst = out_category / "test" / "good"
gt_dst = out_category / "ground_truth" / "good"

train_dst.mkdir(parents=True, exist_ok=True)
test_dst.mkdir(parents=True, exist_ok=True)
gt_dst.mkdir(parents=True, exist_ok=True)

for p in train_src.glob("*"):
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        shutil.copy2(p, train_dst / p.name)

for p in test_src.glob("*"):
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        shutil.copy2(p, test_dst / p.name)

dummy = Image.fromarray(np.zeros((256, 256), dtype=np.uint8))
dummy.save(gt_dst / "dummy.png")

print("Готово")
print(f"train: {len(list(train_dst.glob('*')))}")
print(f"test: {len(list(test_dst.glob('*')))}")
print(f"saved: {out_root}")