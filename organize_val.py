from pathlib import Path
from scipy.io import loadmat
import shutil

base = Path("/home/ubuntu/Desktop/hyenapixel/imagenet")
val_dir = base / "val"
devkit = base / "devkit" / "ILSVRC2012_devkit_t12" / "data"

gt_file = devkit / "ILSVRC2012_validation_ground_truth.txt"
meta_file = devkit / "meta.mat"

meta = loadmat(meta_file, squeeze_me=True)["synsets"]

idx_to_wnid = {}
for m in meta:
    # m[0] = ILSVRC2012_ID, m[1] = wnid
    try:
        ilsvrc_id = int(m[0])
        wnid = str(m[1])
        idx_to_wnid[ilsvrc_id] = wnid
    except Exception:
        pass

with open(gt_file, "r") as f:
    gt = [int(x.strip()) for x in f.readlines()]

images = sorted(val_dir.glob("ILSVRC2012_val_*.JPEG"))

if len(images) != len(gt):
    print(f"Număr imagini: {len(images)}, număr etichete: {len(gt)}")
    raise SystemExit("Numărul de imagini din val nu corespunde cu ground truth.")

for img, cls_id in zip(images, gt):
    wnid = idx_to_wnid[cls_id]
    target_dir = val_dir / wnid
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(img), str(target_dir / img.name))

print("Gata. Folderul val a fost organizat pe clase.")