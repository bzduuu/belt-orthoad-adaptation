from pathlib import Path
import shutil
import cv2
import numpy as np


project_dir = Path.cwd()
roi_dir = project_dir / "roi_defect_aligned"
patches_dir = project_dir / "patches_defect_test"
good_dir = patches_dir / "good"
rejected_dir = patches_dir / "rejected"
glare_dir = rejected_dir / "glare"
dark_dir = rejected_dir / "dark"


clear_old_patches = True

if clear_old_patches and patches_dir.exists():
    shutil.rmtree(patches_dir)

for d in [good_dir, glare_dir, dark_dir]:
    d.mkdir(parents=True, exist_ok=True)


patch_size = 512
stride = 256

glare_ratio_limit = 0.08
dark_mean_limit = 8


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, image):
    ok, buf = cv2.imencode(path.suffix, image)
    if ok:
        buf.tofile(str(path))
    return ok


def get_patch_quality(patch):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    glare_mask = (v > 245) | ((v > 225) & (s < 80))
    glare_ratio = glare_mask.mean()

    mean_v = v.mean()

    if glare_ratio > glare_ratio_limit:
        return "glare", glare_ratio, mean_v

    if mean_v < dark_mean_limit:
        return "dark", glare_ratio, mean_v

    return "good", glare_ratio, mean_v


total = 0
saved_good = 0
saved_glare = 0
saved_dark = 0
skipped_small = 0

for image_path in sorted(roi_dir.glob("*")):
    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    image = imread_unicode(image_path)
    if image is None:
        print(f"Не удалось открыть: {image_path.name}")
        continue

    h, w = image.shape[:2]

    if h < patch_size or w < patch_size:
        print(f"Слишком маленькая ROI для патчей {patch_size}x{patch_size}: {image_path.name}, размер {w}x{h}")
        skipped_small += 1
        continue

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            quality, glare_ratio, mean_v = get_patch_quality(patch)

            file_name = f"{image_path.stem}_y{y:04d}_x{x:04d}_glare{glare_ratio:.3f}_v{mean_v:.1f}.png"

            if quality == "good":
                out_path = good_dir / file_name
                saved_good += 1
            elif quality == "glare":
                out_path = glare_dir / file_name
                saved_glare += 1
            else:
                out_path = dark_dir / file_name
                saved_dark += 1

            imwrite_unicode(out_path, patch)
            total += 1

print(f"Всего патчей: {total}")
print(f"Хороших патчей: {saved_good}")
print(f"Отклонено из-за бликов: {saved_glare}")
print(f"Отклонено из-за темноты: {saved_dark}")
print(f"Пропущено маленьких ROI: {skipped_small}")
print(f"Папка с хорошими патчами: {good_dir}")