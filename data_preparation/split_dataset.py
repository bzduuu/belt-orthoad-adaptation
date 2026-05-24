from pathlib import Path
import shutil
import random


project_dir = Path.cwd()

source_dir = project_dir / "patches_all" / "good"
dataset_dir = project_dir / "belt_dataset_all"

train_dir = dataset_dir / "train" / "good"
test_dir = dataset_dir / "test" / "good"

if dataset_dir.exists():
    shutil.rmtree(dataset_dir)

train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

images = sorted([
    p for p in source_dir.glob("*")
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
])

groups = {}

for p in images:
    name = p.stem
    if "_y" in name:
        group_name = name.split("_y")[0]
    else:
        group_name = name

    groups.setdefault(group_name, []).append(p)

group_names = sorted(groups.keys())

random.seed(42)
random.shuffle(group_names)

test_share = 0.1
test_count = max(1, int(len(group_names) * test_share))

test_groups = set(group_names[:test_count])
train_groups = set(group_names[test_count:])

train_count = 0
test_count_files = 0

for group_name, files in groups.items():
    if group_name in test_groups:
        target_dir = test_dir
    else:
        target_dir = train_dir

    for src in files:
        dst = target_dir / src.name
        shutil.copy2(src, dst)

        if group_name in test_groups:
            test_count_files += 1
        else:
            train_count += 1

print(f"Всего патчей: {len(images)}")
print(f"Групп ROI: {len(group_names)}")
print(f"Train ROI-групп: {len(train_groups)}")
print(f"Test ROI-групп: {len(test_groups)}")
print(f"Train патчей: {train_count}")
print(f"Test патчей: {test_count_files}")
print(f"Датасет сохранен: {dataset_dir}")