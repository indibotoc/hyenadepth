"""Download KITTI raw data for monodepth2 eigen_zhou split.

Downloads each zip from kitti_archives_to_download.txt sequentially,
extracts it into kitti_data/, then deletes the zip to save space.
Skips files that are already extracted. Logs progress to download_kitti.log.
"""
import os
import subprocess
import zipfile
import sys
import time

KITTI_DATA_DIR = "/home/ubuntu/Desktop/hyenapixel/kitti_data"
ARCHIVE_LIST   = "/home/ubuntu/Desktop/hyenapixel/monodepth2/splits/kitti_archives_to_download.txt"
LOG_FILE       = "/home/ubuntu/Desktop/hyenapixel/download_kitti.log"
WGET           = "/usr/bin/wget"

os.makedirs(KITTI_DATA_DIR, exist_ok=True)

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def already_extracted(url):
    """Check if the drive folder for this url already exists in kitti_data."""
    # url like: .../2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip
    # or:       .../2011_09_26_calib.zip
    fname = url.split("/")[-1]          # e.g. 2011_09_26_drive_0001_sync.zip
    name  = fname.replace(".zip", "")   # e.g. 2011_09_26_drive_0001_sync

    # For calib files: check for calib_cam_to_cam.txt in the date folder
    if "calib" in fname and "drive" not in fname:
        date = fname[:10]               # e.g. 2011_09_26
        marker = os.path.join(KITTI_DATA_DIR, date, "calib_cam_to_cam.txt")
        return os.path.exists(marker)

    # For drive syncs: check for the sync folder
    date = name[:10]
    folder = os.path.join(KITTI_DATA_DIR, date, name)
    return os.path.isdir(folder)

def download_and_extract(url):
    fname = url.split("/")[-1]
    zip_path = os.path.join(KITTI_DATA_DIR, fname)

    # Download with resume support (-c)
    log(f"Downloading {fname} ...")
    result = subprocess.run(
        [WGET, "-c", "-q", "--show-progress", "-P", KITTI_DATA_DIR, url],
        capture_output=False
    )
    if result.returncode != 0:
        log(f"  ERROR downloading {fname} (code {result.returncode})")
        return False

    # Extract
    log(f"Extracting {fname} ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(KITTI_DATA_DIR)
    except Exception as e:
        log(f"  ERROR extracting {fname}: {e}")
        return False

    # Delete zip
    os.remove(zip_path)
    log(f"  Done. Zip deleted.")
    return True

with open(ARCHIVE_LIST) as f:
    urls = [l.strip() for l in f if l.strip()]

log(f"Starting KITTI download: {len(urls)} archives -> {KITTI_DATA_DIR}")

ok = skip = fail = 0
for i, url in enumerate(urls, 1):
    fname = url.split("/")[-1]
    if already_extracted(url):
        log(f"[{i}/{len(urls)}] SKIP (already extracted): {fname}")
        skip += 1
        continue
    log(f"[{i}/{len(urls)}] START: {fname}")
    if download_and_extract(url):
        ok += 1
    else:
        fail += 1

log(f"Finished. ok={ok} skipped={skip} failed={fail}")
