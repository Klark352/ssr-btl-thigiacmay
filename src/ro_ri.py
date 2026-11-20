"""Đo mức rò rỉ dữ liệu giữa các tập train, val, test (chạy trên CPU, vài phút).

Hai phép đo:
1. Trùng ảnh gốc: tên file Roboflow có dạng <ảnh gốc>_jpg.rf.<mã>. Các bản tăng cường của cùng một ảnh gốc
   mà nằm ở hai tập khác nhau thì kết quả kiểm định bị lạc quan.
2. Giống nhau về hình ảnh: tính dHash 64 bit cho mỗi ảnh và đo khoảng cách Hamming từ mỗi ảnh val, test tới
   ảnh giống nhất trong train. Khoảng cách từ 0 đến 5 xem là gần như trùng, từ 6 đến 10 là rất giống.
   Phép đo này bắt được cả ảnh chụp liên tiếp, nhưng không bắt được ảnh bị xoay.

Dùng:  import ro_ri; ro_ri.phan_tich()
"""
import glob
import json
import os
import re

import numpy as np
from PIL import Image

DATA = "/kaggle/working/ssr"
LOGS = "/kaggle/working/logs"
ANH = (".jpg", ".jpeg", ".png", ".bmp")
NGUONG_TRUNG = 5      # khoảng cách Hamming tối đa để coi là gần như trùng
NGUONG_GIONG = 10     # khoảng cách Hamming tối đa để coi là rất giống


def anh_goc(ten_tep):
    """Tên ảnh gốc (bỏ phần mã Roboflow và hậu tố của bản tăng cường)."""
    ten = os.path.splitext(os.path.basename(ten_tep))[0]
    ten = re.split(r"_jpg\.rf\.|\.rf\.", ten)[0]
    return re.sub(r"_(rot|flip|aug)\w*$", "", ten)


def dhash(duong_dan):
    """dHash 64 bit: thu ảnh về 9x8 xám rồi so sánh điểm ảnh kề nhau theo hàng."""
    with Image.open(duong_dan) as a:
        a = a.convert("L").resize((9, 8), Image.LANCZOS)
    p = np.asarray(a, dtype=np.int16)
    return (p[:, 1:] > p[:, :-1]).flatten()


def phan_tich(data=DATA, so_cap_hien_thi=5):
    tep = {t: sorted(f for f in glob.glob(f"{data}/images/{t}/*") if f.lower().endswith(ANH))
           for t in ("train", "val", "test")}
    goc = {t: [anh_goc(f) for f in tep[t]] for t in tep}
    print("Số ảnh:", {t: len(v) for t, v in tep.items()})
    print("Ví dụ tên ảnh gốc suy ra:", [(os.path.basename(f)[:28], anh_goc(f)) for f in tep["train"][:3]])
    print("Số ảnh gốc khác nhau:", {t: len(set(v)) for t, v in goc.items()})
    print("Tính dHash cho mọi ảnh ...")
    ham = {t: np.array([dhash(f) for f in tep[t]]) for t in tep}

    ket_qua = {}
    goc_train = set(goc["train"])
    for t in ("val", "test"):
        n = len(tep[t])
        trung_ten = sum(g in goc_train for g in goc[t])
        kc = (ham[t][:, None, :] != ham["train"][None, :, :]).sum(-1)    # ma trận khoảng cách tới ảnh train
        gan_nhat, vi_tri = kc.min(1), kc.argmin(1)
        ket_qua[t] = {
            "so_anh": n,
            "trung_anh_goc_voi_train": int(trung_ten),
            "gan_nhu_trung_(<=%d)" % NGUONG_TRUNG: int((gan_nhat <= NGUONG_TRUNG).sum()),
            "rat_giong_(<=%d)" % NGUONG_GIONG: int((gan_nhat <= NGUONG_GIONG).sum()),
            "khoang_cach_trung_vi": float(np.median(gan_nhat)),
        }
        print(f"\n== {t} so với train ({n} ảnh) ==")
        print(f"  Cùng ảnh gốc với một ảnh train      : {trung_ten} ({trung_ten / n:.1%})")
        print(f"  Gần như trùng (khoảng cách <= {NGUONG_TRUNG})      : {(gan_nhat <= NGUONG_TRUNG).sum()} ({(gan_nhat <= NGUONG_TRUNG).mean():.1%})")
        print(f"  Rất giống (khoảng cách <= {NGUONG_GIONG})         : {(gan_nhat <= NGUONG_GIONG).sum()} ({(gan_nhat <= NGUONG_GIONG).mean():.1%})")
        print(f"  Khoảng cách trung vị tới ảnh train gần nhất: {np.median(gan_nhat):.0f} (tối đa 64)")
        print(f"  {so_cap_hien_thi} cặp giống nhất:")
        for i in np.argsort(gan_nhat)[:so_cap_hien_thi]:
            print(f"    {os.path.basename(tep[t][i])[:34]}  <->  {os.path.basename(tep['train'][vi_tri[i]])[:34]}  (khoảng cách {gan_nhat[i]})")

    chung_vt = set(goc["val"]) & set(goc["test"])
    ket_qua["val_voi_test_cung_anh_goc"] = len(chung_vt)
    print(f"\nẢnh gốc xuất hiện ở cả val và test: {len(chung_vt)}")

    os.makedirs(LOGS, exist_ok=True)
    with open(f"{LOGS}/ro_ri.json", "w", encoding="utf-8") as f:
        json.dump(ket_qua, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu {LOGS}/ro_ri.json")
    return ket_qua


if __name__ == "__main__":
    phan_tich()
