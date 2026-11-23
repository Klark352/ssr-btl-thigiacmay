"""Các bước đánh giá cần PyTorch và GPU: độ phức tạp, đánh giá trên tập test, ảnh dự đoán.

Chạy trong một notebook Kaggle RIÊNG (không chạy chung phiên với huấn luyện), sau khi đã có best.pt của các thí nghiệm.
Yêu cầu: đã chạy ssr_lib.chuan_bi_du_lieu() và ssr_lib.tao_yaml_du_lieu(); repo gốc YOLO-SSR nằm ở ssr_lib.REPO.
Các hàm dùng tiến trình con (giống lúc huấn luyện) nên tiến trình chính của notebook không khởi tạo CUDA.

Lưu ý: mã gọi vào API của nhánh Ultralytics 8.0.105 đa nhiệm của tác giả. Chưa kiểm thử trên repo thật,
nên mỗi hàm đều in rõ lỗi và có cách kiểm tra chéo bằng số liệu của lần huấn luyện.
"""
import glob
import json
import multiprocessing as mp
import os
import sys
import time

import ssr_lib as L
import ket_qua as K


# ---------------------------------------------------------------- 1. ĐỘ PHỨC TẠP: tham số, GFLOPs, độ trễ
def _nap_mo_hinh(best):
    """Nạp mô hình từ best.pt về CPU (cần repo gốc trong sys.path để giải mã các lớp tự định nghĩa)."""
    import torch
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    if L.REPO not in sys.path:
        sys.path.insert(0, L.REPO)
    ck = torch.load(best, map_location="cpu", weights_only=False)
    m = (ck.get("ema") or ck["model"]).float().eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def _so_tham_so(m):
    return sum(p.numel() for p in m.parameters())


def _gflops(m, imgsz):
    """GFLOPs cho một ảnh imgsz×imgsz. Thử hàm của Ultralytics, nếu lỗi thì dùng thop."""
    import torch
    try:
        from ultralytics.yolo.utils.torch_utils import get_flops
        return float(get_flops(m, imgsz))
    except Exception as loi:
        print("  get_flops lỗi, thử thop:", repr(loi)[:120])
    try:
        from thop import profile
        x = torch.zeros(1, 3, imgsz, imgsz)
        return float(profile(m, inputs=(x,), verbose=False)[0] / 1e9 * 2)
    except Exception as loi:
        print("  thop lỗi:", repr(loi)[:120])
        return None


def _do_tre(m, imgsz, thiet_bi, lan=100, khoi_dong=15):
    """Độ trễ trung bình (ms) cho một ảnh (batch = 1), chỉ tính lan truyền thuận, không tính tiền và hậu xử lý."""
    import torch
    m = m.to(thiet_bi).eval()
    x = torch.zeros(1, 3, imgsz, imgsz, device=thiet_bi)
    dong_bo = (lambda: torch.cuda.synchronize()) if thiet_bi == "cuda" else (lambda: None)
    with torch.no_grad():
        for _ in range(khoi_dong):
            m(x)
        dong_bo()
        t0 = time.perf_counter()
        for _ in range(lan):
            m(x)
        dong_bo()
    return (time.perf_counter() - t0) / lan * 1000


def do_phuc_tap(ds, imgsz=640, lan=100, gpu=True, cpu=True, luu_vao=None):
    """Đo số tham số, GFLOPs và độ trễ của từng thí nghiệm có best.pt. Trả về DataFrame (lưu CSV nếu có luu_vao)."""
    import pandas as pd
    import torch
    dong = []
    for ten, f in ds.items():
        if not f.get("best"):
            print(f"[{ten}] không có best.pt, bỏ qua")
            continue
        print(f"[{ten}] đang đo ...")
        try:
            m = _nap_mo_hinh(f["best"])
            r = {"Mã": ten, "Số tham số (khi huấn luyện)": _so_tham_so(m), "GFLOPs": _gflops(m, imgsz)}
            if hasattr(m, "fuse"):
                try:
                    m = m.fuse()
                    r["Số tham số (sau hợp nhất BN)"] = _so_tham_so(m)
                except Exception as loi:
                    print("  fuse lỗi:", repr(loi)[:120])
            if cpu:
                torch.set_num_threads(max(1, os.cpu_count() or 1))
                r["Độ trễ CPU (ms)"] = round(_do_tre(m, imgsz, "cpu", lan=max(10, lan // 5)), 2)
            if gpu and torch.cuda.is_available():
                r["Độ trễ GPU (ms)"] = round(_do_tre(m, imgsz, "cuda", lan=lan), 2)
                r["FPS GPU"] = round(1000 / r["Độ trễ GPU (ms)"], 1)
            dong.append(r)
            print("  ", {k: v for k, v in r.items() if k != "Mã"})
        except Exception as loi:
            print(f"[{ten}] LỖI: {loi!r}")
    b = pd.DataFrame(dong)
    if luu_vao and len(b):
        os.makedirs(luu_vao, exist_ok=True)
        b.to_csv(f"{luu_vao}/phuc_tap.csv", index=False, encoding="utf-8-sig")
    return b


# ---------------------------------------------------------------- 2. ĐÁNH GIÁ TRÊN TẬP TEST
def _val_worker(best, kw, tep_log):
    fd = os.open(tep_log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    sys.stdout = os.fdopen(1, "w", buffering=1, closefd=False)
    sys.stderr = os.fdopen(2, "w", buffering=1, closefd=False)
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ["WANDB_DISABLED"] = "true"
    for m in [m for m in sys.modules if m == "ultralytics" or m.startswith("ultralytics.")]:
        del sys.modules[m]
    for ten in L.THU_VIEN_AN:
        sys.modules[ten] = None
    sys.path.insert(0, L.REPO)
    os.chdir(L.REPO)
    from ultralytics import YOLO
    YOLO(best, task="multi").val(**kw)


def danh_gia(ten, best, split="test", imgsz=640, batch=16):
    """Chạy kiểm định best.pt trên tập split ('test' hoặc 'val'). Trả về (phát hiện, phân đoạn) lần cuối trong log.

    Cách kiểm tra chéo: chạy split='val' trước. Con số phải khớp lần kiểm định cuối trong log huấn luyện;
    khi khớp thì mới tin kết quả split='test'.
    """
    os.makedirs(L.LOGS, exist_ok=True)
    tep_log = f"{L.LOGS}/{split}_{ten}.log"
    kw = dict(data=L.YAML_DU_LIEU, task="multi", split=split, imgsz=imgsz, batch=batch, device=[0],
              workers=2, plots=False, project=f"{L.RUNS}/_danh_gia", name=f"{split}_{ten}", exist_ok=True)
    p = mp.get_context("fork").Process(target=_val_worker, args=(best, kw, tep_log))
    print(f"[{ten}] đánh giá trên {split} ...")
    p.start(); p.join()
    if p.exitcode != 0:
        print(f"[{ten}] LỖI (mã {p.exitcode}). 30 dòng cuối của log:")
        print("\n".join(open(tep_log, encoding="utf-8", errors="ignore").read().splitlines()[-30:]))
        return None, None
    det, seg = K.doc_log(tep_log)
    return (det[-1] if det else None), (seg[-1] if seg else None)


def danh_gia_tat_ca(ds, split="test", luu_vao=None):
    """Đánh giá mọi thí nghiệm có best.pt trên tập split và gom thành một bảng cùng định dạng với ket_qua.bang_tong_hop."""
    import pandas as pd
    dong = []
    for ten, f in ds.items():
        if not f.get("best"):
            continue
        d, s = danh_gia(ten, f["best"], split=split)
        r = {"Mã": ten, "Tập": split}
        if d:
            r.update({"P": d[2], "R": d[3], "mAP50": d[4], "mAP50-95": d[5]})
        if s:
            r.update({"pixacc": s[0], "IoU": s[2], "mIoU": s[3]})
        dong.append(r)
        print("  ", {k: v for k, v in r.items() if k not in ("Mã", "Tập")})
    b = pd.DataFrame(dong)
    if luu_vao and len(b):
        os.makedirs(luu_vao, exist_ok=True)
        b.to_csv(f"{luu_vao}/{split}.csv", index=False, encoding="utf-8-sig")
    return b


def kiem_tra_chay_val(ds, ten="E_main"):
    """So khớp lần kiểm định lại trên tập val với số cuối trong log huấn luyện, để chắc rằng cách gọi đúng."""
    f = ds[ten]
    d_log, s_log = K.doc_log(f["log"])
    d, s = danh_gia(ten, f["best"], split="val")
    print("Log huấn luyện (cuối): phát hiện", d_log[-1] if d_log else None, "| phân đoạn", s_log[-1] if s_log else None)
    print("Chạy lại trên val     : phát hiện", d, "| phân đoạn", s)
    if d and d_log:
        lech = max(abs(a - b) for a, b in zip(d[2:], d_log[-1][2:]))
        print("Độ lệch lớn nhất của P, R, mAP:", round(lech, 4), "(nhỏ hơn 0,01 là khớp)")


# ---------------------------------------------------------------- 3. ẢNH DỰ ĐOÁN VÀ CA KHÓ
def chon_anh_kho(so_anh=12, tap="test"):
    """Chọn ảnh của tập tap có nhiều thân cây nhất (dày đặc thì khó) và một số ảnh ngẫu nhiên, dựa trên nhãn phát hiện."""
    import random
    nhan = glob.glob(f"{L.DATA}/detection-object/labels/{tap}/*.txt")
    dem = []
    for f in nhan:
        n = sum(1 for d in open(f, encoding="utf-8") if d.strip())
        dem.append((n, os.path.splitext(os.path.basename(f))[0]))
    dem.sort(reverse=True)
    anh = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(f"{L.DATA}/images/{tap}/*.*")}
    kho = [anh[t] for _, t in dem[: so_anh // 2] if t in anh]
    con_lai = [p for t, p in anh.items() if p not in kho]
    random.Random(0).shuffle(con_lai)
    return kho + con_lai[: so_anh - len(kho)]


def _pred_worker(best, kw, tep_log):
    fd = os.open(tep_log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 1); os.dup2(fd, 2)
    sys.stdout = os.fdopen(1, "w", buffering=1, closefd=False)
    sys.stderr = os.fdopen(2, "w", buffering=1, closefd=False)
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    for m in [m for m in sys.modules if m == "ultralytics" or m.startswith("ultralytics.")]:
        del sys.modules[m]
    for ten in L.THU_VIEN_AN:
        sys.modules[ten] = None
    sys.path.insert(0, L.REPO)
    os.chdir(L.REPO)
    from ultralytics import YOLO
    YOLO(best, task="multi").predict(**kw)


def du_doan(ten, best, anh, imgsz=640, conf=0.25):
    """Dự đoán trên danh sách ảnh, lưu ảnh kết quả vào runs/_du_doan/<ten>. Trả về thư mục chứa ảnh kết quả."""
    dich = f"{L.RUNS}/_du_doan"
    tep_log = f"{L.LOGS}/du_doan_{ten}.log"
    kw = dict(source=anh, imgsz=imgsz, conf=conf, save=True, project=dich, name=ten, exist_ok=True, device=0)
    p = mp.get_context("fork").Process(target=_pred_worker, args=(best, kw, tep_log))
    p.start(); p.join()
    if p.exitcode != 0:
        print(f"[{ten}] LỖI khi dự đoán. 30 dòng cuối của log:")
        print("\n".join(open(tep_log, encoding="utf-8", errors="ignore").read().splitlines()[-30:]))
        return None
    return f"{dich}/{ten}"


def ghep_luoi(thu_muc, so_cot=3, luu_vao=None, hien=True):
    """Ghép các ảnh kết quả trong thư mục thành một lưới để đưa vào báo cáo."""
    import matplotlib.pyplot as plt
    from PIL import Image
    tep = sorted(glob.glob(f"{thu_muc}/*.jpg") + glob.glob(f"{thu_muc}/*.png"))
    if not tep:
        print("Không có ảnh kết quả trong", thu_muc)
        return None
    hang = (len(tep) + so_cot - 1) // so_cot
    fig, ax = plt.subplots(hang, so_cot, figsize=(4.2 * so_cot, 3.4 * hang))
    ax = ax.ravel() if hasattr(ax, "ravel") else [ax]
    for a in ax:
        a.axis("off")
    for a, t in zip(ax, tep):
        a.imshow(Image.open(t)); a.set_title(os.path.basename(t)[:28], fontsize=8)
    fig.tight_layout()
    if luu_vao:
        os.makedirs(os.path.dirname(luu_vao) or ".", exist_ok=True)
        fig.savefig(luu_vao, dpi=150)
    if hien:
        plt.show()
    return luu_vao
