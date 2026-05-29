from pathlib import Path
import json
import cv2
import numpy as np


project_dir = Path.cwd()
annotations_dir = project_dir / "annotations_defect_roi"
raw_sample_dir = project_dir / "raw_defect_sample"
raw_photos_dir = project_dir / "raw_defect_photos"
output_dir = project_dir / "roi_defect_aligned"

output_dir.mkdir(exist_ok=True)


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, image):
    ext = path.suffix
    ok, buf = cv2.imencode(ext, image)
    if ok:
        buf.tofile(str(path))
    return ok


def order_points(points):
    pts = np.array(points, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def find_image_path(json_path, image_path_from_json):
    candidates = []

    if image_path_from_json:
        candidates.append(json_path.parent / image_path_from_json)
        candidates.append(raw_sample_dir / Path(image_path_from_json).name)
        candidates.append(raw_photos_dir / Path(image_path_from_json).name)

    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        candidates.append(raw_sample_dir / f"{json_path.stem}{ext}")
        candidates.append(raw_photos_dir / f"{json_path.stem}{ext}")

    for path in candidates:
        if path.exists():
            return path

    return None


for json_path in sorted (annotations_dir.glob("*.json")):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_path = find_image_path(json_path, data.get("imagePath"))
    if image_path is None:
        print(f"Не найдено изображение для {json_path.name}")
        continue

    image = imread_unicode(image_path)
    if image is None:
        print(f"Не удалось открыть изображение: {image_path}")
        continue

    belt_shapes = [s for s in data["shapes"] if s["label"] == "belt"]

    if len(belt_shapes) == 0:
        print(f"Нет разметки belt в {json_path.name}")
        continue

    points = belt_shapes[0]["points"]

    if len(points) != 4:
        rect = cv2.minAreaRect(np.array(points, dtype=np.float32))
        points = cv2.boxPoints(rect)

    pts = order_points(points)

    width_top = np.linalg.norm(pts[1] - pts[0])
    width_bottom = np.linalg.norm(pts[2] - pts[3])
    height_right = np.linalg.norm(pts[2] - pts[1])
    height_left = np.linalg.norm(pts[3] - pts[0])

    width = int(max(width_top, width_bottom))
    height = int(max(height_right, height_left))

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))

    out_path = output_dir / f"{json_path.stem}_roi.png"
    imwrite_unicode(out_path, warped)

    print(f"Сохранено: {out_path.name}, размер: {warped.shape[1]}x{warped.shape[0]}")