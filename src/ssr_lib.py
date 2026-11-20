"""Thư viện dùng chung cho mọi thí nghiệm YOLO-SSR trên Kaggle.

Lưu ý quan trọng: notebook (tiến trình chính) KHÔNG được khởi tạo CUDA. Việc huấn luyện chạy trong
một tiến trình con riêng, và file này không import torch ở tiến trình chính.
"""
import glob
import json
import multiprocessing as mp
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

BASE = "/kaggle/working"
REPO = f"{BASE}/YOLO-SSR"
DATA = f"{BASE}/ssr"                      # dữ liệu đã thu nhỏ, dùng để huấn luyện
RUNS = f"{BASE}/runs"
LOGS = f"{BASE}/logs"
YAML_DU_LIEU = f"{BASE}/ssr-multi.yaml"
CONFIGS = f"{BASE}/ssr-btl-thigiacmay/configs"   # nơi chứa các yaml mô hình của riêng bạn
NGUON_GOC = "/kaggle/input/datasets/xxdxdxd/seedlingstemrow/SSR/SSR"
ANH = (".jpg", ".jpeg", ".png", ".bmp")
CANH_DAI = 1280                           # cạnh dài tối đa của ảnh sau khi thu nhỏ
# Thư viện ghi nhật ký mà callback cũ của repo gọi sai trên môi trường mới
THU_VIEN_AN = ["wandb", "ray", "comet_ml", "clearml", "mlflow", "dvc", "neptune",
               "torch.utils.tensorboard", "tensorboard"]


# ---------------------------------------------------------------- DỮ LIỆU
def _xu_ly_mot_tep(nguon, dich):
    if os.path.exists(dich):
        return
    os.makedirs(os.path.dirname(dich), exist_ok=True)
    if not nguon.lower().endswith(ANH):
        shutil.copy2(nguon, dich)         # file nhãn và file khác: chép nguyên
        return
    from PIL import Image, ImageOps
    with Image.open(nguon) as anh:
        anh = ImageOps.exif_transpose(anh).convert("RGB")   # xoay đúng hướng theo thẻ EXIF
        he_so = CANH_DAI / max(anh.size)
        if he_so < 1:
            anh = anh.resize((round(anh.width * he_so), round(anh.height * he_so)), Image.LANCZOS)
        anh.save(dich, quality=95)


def thu_nho_du_lieu(nguon=NGUON_GOC, dich=DATA, so_luong=4):
    """Sao chép cây thư mục dữ liệu, thu nhỏ ảnh về cạnh dài tối đa 1280. Nhãn YOLO đã chuẩn hóa nên giữ nguyên."""
    cap = []
    for goc, _, files in os.walk(nguon):
        for f in files:
            n = os.path.join(goc, f)
            cap.append((n, os.path.join(dich, os.path.relpath(n, nguon))))
    print(f"Thu nhỏ {len(cap)} file từ {nguon} sang {dich} ...")
    t0, xong = time.time(), 0
    with ThreadPoolExecutor(so_luong) as ex:
        for _ in ex.map(lambda c: _xu_ly_mot_tep(*c), cap):
            xong += 1
            if xong % 2000 == 0:
                print(f"  {xong}/{len(cap)} file, {round((time.time() - t0) / 60, 1)} phút")
    print(f"Xong sau {round((time.time() - t0) / 60, 1)} phút.")


def _thu_muc_anh(goc):
    """Tìm các thư mục chứa ảnh và xếp theo tập train, val, test dựa vào tên thư mục."""
    kq = {}
    for d, _, files in os.walk(goc):
        if any(f.lower().endswith(ANH) for f in files):
            phan = [p.lower() for p in os.path.relpath(d, goc).split(os.sep)]
            for ten, khoa in (("train", "train"), ("val", "val"), ("valid", "val"), ("test", "test")):
                if ten in phan:
                    kq.setdefault(khoa, []).append(d)
    return kq


# Cấu hình dữ liệu mặc định, lấy từ ultralytics/datasets/bdd-multi.yaml của chính tác giả YOLO-SSR
# (file đó viết cho đúng bộ dữ liệu này). Nếu repo có file đó thì đọc trực tiếp từ đó.
YAML_MAU = f"{REPO}/ultralytics/datasets/bdd-multi.yaml"
MAC_DINH = {
    "labels_list": ["detection-object", "seg-line"],   # tên thư mục nhãn của từng nhiệm vụ (phát hiện, phân đoạn)
    "tnc": 2,                                          # tổng số lớp
    "nc_list": [1, 1],                                 # số lớp của từng nhiệm vụ
    "map": [{"1": "0"}, {"0": "0"}],                   # đổi id lớp trong file nhãn: phát hiện 1 -> 0, phân đoạn 0 -> 0
    "names": {0: "line", 1: "stem"},                   # 0 = hàng cây, 1 = thân cây
}


def _id_lop_trong_nhan(nhiem_vu, tap):
    """Đếm số file nhãn và các id lớp xuất hiện trong thư mục nhãn của một nhiệm vụ."""
    tep = glob.glob(f"{DATA}/{nhiem_vu}/labels/{tap}/*.txt")
    ids = set()
    for f in tep:
        with open(f, encoding="utf-8") as fh:
            for dong in fh:
                if dong.strip():
                    ids.add(dong.split()[0])
    return len(tep), ids


def tao_yaml_du_lieu():
    """Tạo ssr-multi.yaml theo đúng khuôn của tác giả, trỏ vào dữ liệu đã thu nhỏ, rồi đối chiếu với dữ liệu thật."""
    import yaml
    mau = {k: v for k, v in MAC_DINH.items()}
    nguon = "giá trị mặc định (chép từ bdd-multi.yaml của tác giả)"
    if os.path.exists(YAML_MAU):
        with open(YAML_MAU, encoding="utf-8") as f:
            goc = yaml.safe_load(f)
        for k in MAC_DINH:
            if k in goc:
                mau[k] = goc[k]
        nguon = "ultralytics/datasets/bdd-multi.yaml của tác giả"
    cau_hinh = {"path": DATA, "train": "images/train", "val": "images/val", "test": "images/test", **mau}
    with open(YAML_DU_LIEU, "w", encoding="utf-8") as f:
        yaml.safe_dump(cau_hinh, f, allow_unicode=True, sort_keys=False)
    print(f"Đã tạo {YAML_DU_LIEU} (khuôn lấy từ: {nguon})")
    print("  nhiệm vụ  :", mau["labels_list"], "| nc_list:", mau["nc_list"], "| tnc:", mau["tnc"])
    print("  tên lớp   :", mau["names"], "| map:", mau["map"])

    # Đối chiếu với dữ liệu thật để phát hiện sai lệch ngay từ bước chuẩn bị
    van_de = []
    for tap in ("train", "val", "test"):
        thu_muc = f"{DATA}/images/{tap}"
        so_anh = len([x for x in os.listdir(thu_muc) if x.lower().endswith(ANH)]) if os.path.isdir(thu_muc) else 0
        dong = [f"  {tap:<5}: {so_anh} ảnh"]
        if not so_anh:
            van_de.append(f"không có ảnh trong {thu_muc}")
        for i, nv in enumerate(mau["labels_list"]):
            so_nhan, ids = _id_lop_trong_nhan(nv, tap)
            khoa = set(mau["map"][i].keys()) if isinstance(mau["map"][i], dict) else None
            dong.append(f"{nv}: {so_nhan} file nhãn, id lớp {sorted(ids)}")
            if so_nhan != so_anh:
                van_de.append(f"{tap}/{nv}: {so_nhan} file nhãn nhưng {so_anh} ảnh")
            if khoa is not None and not ids <= khoa:
                van_de.append(f"{tap}/{nv}: id lớp {sorted(ids)} không nằm trong map {sorted(khoa)}")
        print(" | ".join(dong))
    if van_de:
        print("CẢNH BÁO, cần xử lý trước khi huấn luyện:")
        for v in van_de:
            print("  -", v)
    else:
        print("Đối chiếu ảnh, nhãn và map: khớp.")


def chuan_bi_du_lieu(nguon_da_thu_nho=None):
    """Chuẩn bị dữ liệu huấn luyện. Ưu tiên: dùng sẵn thư mục có sẵn > chép từ Kaggle Dataset đã thu nhỏ > thu nhỏ từ dữ liệu gốc."""
    if os.path.isdir(DATA) and _thu_muc_anh(DATA):
        print("Dữ liệu đã thu nhỏ có sẵn:", DATA)
    elif nguon_da_thu_nho and os.path.isdir(nguon_da_thu_nho):
        print("Chép dữ liệu đã thu nhỏ từ", nguon_da_thu_nho)
        shutil.copytree(nguon_da_thu_nho, DATA, dirs_exist_ok=True)
    else:
        thu_nho_du_lieu()
    tao_yaml_du_lieu()


# ---------------------------------------------------------------- MÔI TRƯỜNG VÀ NHẬT KÝ
def _chay(lenh):
    try:
        return subprocess.run(lenh, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as loi:
        return f"(không chạy được: {loi})"


def ghi_moi_truong(ten_tep="moi_truong.txt"):
    """Ghi phiên bản Python, thư viện, GPU và commit của repo gốc để chứng minh khả năng tái lập."""
    os.makedirs(LOGS, exist_ok=True)
    torch_info = _chay([sys.executable, "-c",
                        "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,"
                        "'gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"])
    dong = [f"Thời điểm: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Python: {platform.python_version()} | Hệ điều hành: {platform.platform()}",
            torch_info,
            f"GPU: {_chay(['nvidia-smi', '-L'])}",
            f"Commit repo gốc YOLO-SSR: {_chay(['git', '-C', REPO, 'rev-parse', 'HEAD'])}",
            "---- pip freeze ----", _chay([sys.executable, "-m", "pip", "freeze"])]
    p = os.path.join(LOGS, ten_tep)
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(dong))
    print("\n".join(dong[:5]))
    print(f"Đã ghi đầy đủ vào {p}")


# ---------------------------------------------------------------- HUẤN LUYỆN
def _tim_cau_hinh(mo_hinh):
    """Đổi tên file cấu hình (vd YOLO-SSR.yaml) thành đường dẫn đầy đủ trong repo."""
    if os.path.isfile(mo_hinh):
        return os.path.abspath(mo_hinh)
    for goc in (CONFIGS, REPO):          # ưu tiên yaml của bạn, rồi tới yaml trong repo gốc
        khop = glob.glob(f"{goc}/**/{mo_hinh}", recursive=True)
        if khop:
            return khop[0]
    return mo_hinh


def _train_worker(mo_hinh, kw, tep_log):
    """Chạy trong tiến trình con: ghi toàn bộ đầu ra vào log, nạp lại ultralytics sạch rồi huấn luyện."""
    fd = os.open(tep_log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    sys.stdout = os.fdopen(1, "w", buffering=1, closefd=False)
    sys.stderr = os.fdopen(2, "w", buffering=1, closefd=False)
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ["WANDB_DISABLED"] = "true"
    for m in [m for m in sys.modules if m == "ultralytics" or m.startswith("ultralytics.")]:
        del sys.modules[m]
    for ten in THU_VIEN_AN:
        sys.modules[ten] = None
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    from ultralytics import YOLO
    YOLO(_tim_cau_hinh(mo_hinh), task="multi").train(**kw)


def _tien_do_tu_log(tep_log):
    try:
        with open(tep_log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 6000))
            cuoi = f.read().decode("utf-8", "ignore")
    except OSError:
        return ""
    khop = re.findall(r"\s(\d+)/(\d+)\s+\d+\.?\d*G", cuoi)
    return f"epoch {khop[-1][0]}/{khop[-1][1]}" if khop else ""


def run_train(ten, mo_hinh, tham_so=None):
    """Huấn luyện một thí nghiệm. Trả về True nếu thành công. Log ở /kaggle/working/logs/<ten>.log."""
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(RUNS, exist_ok=True)
    tep_log = f"{LOGS}/{ten}.log"
    kw = dict(data=YAML_DU_LIEU, task="multi", epochs=30, batch=16, imgsz=640, device=[0], workers=4,
              classes=[0, 1], single_cls=False, augment=True, amp=False, cache=False, val=True,
              project=RUNS, name=ten, exist_ok=True, seed=0, deterministic=True)
    kw.update(tham_so or {})
    print(f"[{ten}] bắt đầu: mô hình={mo_hinh}, epochs={kw['epochs']}, imgsz={kw['imgsz']}, "
          f"batch={kw['batch']}, seed={kw['seed']}")
    tien_trinh = mp.get_context("fork").Process(target=_train_worker, args=(mo_hinh, kw, tep_log))
    t0 = time.time()
    tien_trinh.start()
    while tien_trinh.is_alive():
        tien_trinh.join(60)
        print(f"[{ten}] đã chạy {round((time.time() - t0) / 60, 1)} phút {_tien_do_tu_log(tep_log)}", flush=True)
    ok = tien_trinh.exitcode == 0
    print(f"[{ten}] {'HOÀN THÀNH' if ok else 'LỖI'} sau {round((time.time() - t0) / 60, 1)} phút.")
    if not ok:
        with open(tep_log, encoding="utf-8", errors="ignore") as f:
            print("---- 40 dòng cuối của log ----")
            print("".join(f.readlines()[-40:]))
    return ok


def tai_cau_hinh(duong_dan):
    with open(duong_dan, encoding="utf-8") as f:
        return json.load(f)


def chay_thi_nghiem(ma, cau_hinh, epochs=None, ten=None):
    """Chạy thí nghiệm theo mã (E0, E_main, E1, ...) trong configs/experiments.json."""
    tn = cau_hinh["thi_nghiem"][ma]
    mo_hinh = tn.get("trong_so") or tn.get("cfg")
    if not mo_hinh:
        raise ValueError(f"Thí nghiệm {ma} chưa điền 'cfg' (hoặc 'trong_so') trong experiments.json.")
    tham_so = {**cau_hinh["chung"], **tn.get("tham_so", {})}
    if epochs:
        tham_so["epochs"] = epochs
    print(f"{ma}: {tn.get('mo_ta', '')}")
    return run_train(ten or ma, mo_hinh, tham_so)


# ---------------------------------------------------------------- KẾT QUẢ
def tom_tat(ten):
    """In dòng cuối của results.csv (số liệu epoch cuối) và trả về bảng đầy đủ."""
    import pandas as pd
    p = f"{RUNS}/{ten}/results.csv"
    if not os.path.exists(p):
        print("Chưa có", p)
        return None
    bang = pd.read_csv(p)
    bang.columns = [c.strip() for c in bang.columns]
    print(f"[{ten}] {len(bang)} epoch. Số liệu epoch cuối:")
    print(bang.tail(1).T.to_string())
    return bang


def nen_ket_qua(ten):
    """Nén thư mục chạy, log và thông tin môi trường thành một file zip để tải về."""
    dich = f"{BASE}/{ten}.zip"
    with zipfile.ZipFile(dich, "w", zipfile.ZIP_DEFLATED) as z:
        for goc, _, files in os.walk(f"{RUNS}/{ten}"):
            for f in files:
                n = os.path.join(goc, f)
                z.write(n, os.path.join("runs", ten, os.path.relpath(n, f"{RUNS}/{ten}")))
        for f in (f"{LOGS}/{ten}.log", f"{LOGS}/moi_truong.txt"):
            if os.path.exists(f):
                z.write(f, os.path.join("logs", os.path.basename(f)))
    print(f"Đã nén: {dich} ({os.path.getsize(dich) / 1e6:.1f} MB)")
    return dich
