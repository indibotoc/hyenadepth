"""Download KITTI raw data for monodepth2 eigen_zhou split.

Downloads each zip sequentially using urllib (no wget/curl needed),
extracts it into kitti_data/, then deletes the zip to save space.
Skips files that are already extracted. Logs progress to download_kitti.log.
"""
import os
import zipfile
import time
import urllib.request

KITTI_DATA_DIR = "/home/ubuntu/Desktop/hyenapixel/kitti_data"
ARCHIVE_LIST   = "/home/ubuntu/Desktop/hyenapixel/monodepth2/splits/kitti_archives_to_download.txt"
LOG_FILE       = "/home/ubuntu/Desktop/hyenapixel/download_kitti.log"

os.makedirs(KITTI_DATA_DIR, exist_ok=True)


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def already_extracted(url):
    fname = url.split("/")[-1]
    name  = fname.replace(".zip", "")
    if "calib" in fname and "drive" not in fname:
        date = fname[:10]
        return os.path.exists(os.path.join(KITTI_DATA_DIR, date, "calib_cam_to_cam.txt"))
    date   = name[:10]
    folder = os.path.join(KITTI_DATA_DIR, date, name)
    return os.path.isdir(folder)


def download_and_extract(url):
    fname    = url.split("/")[-1]
    zip_path = os.path.join(KITTI_DATA_DIR, fname)

    # Resumable download via urllib
    log(f"  Downloading {fname} ...")
    headers = {}
    if os.path.exists(zip_path):
        done = os.path.getsize(zip_path)
        headers["Range"] = f"bytes={done}-"
        log(f"  Resuming from {done/1e6:.1f} MB")

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, \
             open(zip_path, "ab") as out:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1 << 20  # 1 MB
            t0 = time.time()
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                downloaded += len(buf)
                if time.time() - t0 > 30:
                    mb = downloaded / 1e6
                    log(f"  ... {mb:.0f} MB / {total/1e6:.0f} MB")
                    t0 = time.time()
    except Exception as e:
        log(f"  ERROR downloading {fname}: {e}")
        return False

    log(f"  Extracting {fname} ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(KITTI_DATA_DIR)
    except Exception as e:
        log(f"  ERROR extracting {fname}: {e}")
        return False

    os.remove(zip_path)
    log(f"  Done.")
    return True


with open(ARCHIVE_LIST) as f:
    urls = [l.strip() for l in f if l.strip()]

log(f"Starting KITTI download: {len(urls)} archives -> {KITTI_DATA_DIR}")

ok = skip = fail = 0
for i, url in enumerate(urls, 1):
    fname = url.split("/")[-1]
    if already_extracted(url):
        log(f"[{i}/{len(urls)}] SKIP: {fname}")
        skip += 1
        continue
    log(f"[{i}/{len(urls)}] START: {fname}")
    if download_and_extract(url):
        ok += 1
    else:
        fail += 1

log(f"Finished. ok={ok} skipped={skip} failed={fail}")
