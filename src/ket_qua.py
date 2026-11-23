"""Đọc, tổng hợp và vẽ kết quả của các thí nghiệm E0, E_main, E1, E2, E3 sau khi chạy xong.

Module này chỉ dùng pandas và matplotlib, không cần PyTorch hay GPU, nên chạy được cả trên Kaggle lẫn máy cá nhân.
Đầu vào là thư mục chứa kết quả đã tải về (các file zip do ssr_lib.nen_ket_qua tạo, hoặc thư mục đã giải nén).
Cấu trúc mỗi lần chạy: runs/<tên>/results.csv, runs/<tên>/weights/best.pt và logs/<tên>.log.
"""
import glob
import json
import os
import re
import zipfile

import pandas as pd

THU_TU = ["E0", "E_main", "E1", "E2", "E3"]
MO_TA = {
    "E0": "Baseline (C2f)",
    "E_main": "YOLO-SSR",
    "E1": "YOLO-SSR, lr0 = 0,005",
    "E2": "YOLO-SSR, DGST → C2f",
    "E3": "Fine-tune từ E_main",
}
# Các cặp so sánh dùng ở mục 3.4 và chương VI của báo cáo: (mô hình, mô hình đối chứng, yếu tố khác nhau)
CAC_CAP = [
    ("E_main", "E0", "TriPAC + DGST + SPDC so với C2f"),
    ("E_main", "E2", "DGST so với C2f"),
    ("E_main", "E1", "lr0 = 0,01 so với 0,005"),
]
CHI_SO = ["P", "R", "mAP50", "mAP50-95", "pixacc", "IoU", "mIoU"]
SO = r"([-+\d.eE]+)"


# ------------------------------------------------------------------ tìm và nạp dữ liệu
def giai_nen_va_tim(goc, giai_nen_vao=None):
    """Giải nén các file <tên>.zip trong thư mục goc (nếu có), rồi tìm kết quả của từng thí nghiệm.

    Trả về dict: tên thí nghiệm -> {"csv": đường dẫn results.csv, "log": đường dẫn log hoặc None, "best": best.pt hoặc None}.
    """
    giai_nen_vao = giai_nen_vao or os.path.join(goc, "_giai_nen")      # trên Kaggle, Input chỉ đọc nên cần truyền thư mục ghi được
    for z in sorted(glob.glob(os.path.join(goc, "**", "*.zip"), recursive=True)):
        ten = os.path.splitext(os.path.basename(z))[0]
        dich = os.path.join(giai_nen_vao, ten)
        if ten in THU_TU and not os.path.isdir(dich):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dich)
            print(f"Đã giải nén {z} -> {dich}")
    kq = {}
    csv_tim = []
    for g in dict.fromkeys([goc, giai_nen_vao]):
        csv_tim += glob.glob(os.path.join(g, "**", "runs", "*", "results.csv"), recursive=True)
    for csv in csv_tim:
        ten = os.path.basename(os.path.dirname(csv))
        if ten not in THU_TU or ten in kq:          # bỏ tên lạ và bản trùng (giữ bản tìm thấy đầu tiên)
            continue
        thu_muc_chay = os.path.dirname(csv)
        cha = os.path.dirname(os.path.dirname(thu_muc_chay))       # thư mục chứa runs/ và logs/
        log = os.path.join(cha, "logs", f"{ten}.log")
        best = os.path.join(thu_muc_chay, "weights", "best.pt")
        kq[ten] = {"csv": csv, "log": log if os.path.exists(log) else None, "best": best if os.path.exists(best) else None}
    kq = {k: kq[k] for k in THU_TU if k in kq}
    print("Tìm thấy:", ", ".join(kq) if kq else "chưa có thí nghiệm nào")
    thieu = [k for k in THU_TU[:4] if k not in kq]
    if thieu:
        print("Chưa có kết quả của:", ", ".join(thieu))
    return kq


def doc_log(duong_dan):
    """Lấy toàn bộ số liệu kiểm định theo từng epoch trong log.

    Trả về (det, seg): det là danh sách (ảnh, đối tượng, P, R, mAP50, mAP50-95) và seg là danh sách (pixacc, subacc, IoU, mIoU),
    mỗi phần tử ứng với một lần kiểm định. Phần tử CUỐI là lần kiểm định cuối cùng (với best.pt).
    """
    if not duong_dan or not os.path.exists(duong_dan):
        return [], []
    log = open(duong_dan, encoding="utf-8", errors="ignore").read().replace("\r", "\n")
    det = re.findall(r"^\s*all\s+(\d+)\s+(\d+)\s+" + r"\s+".join([SO] * 4), log, flags=re.M)
    seg = re.findall(r"\bpixacc\s+" + SO + r"\s+subacc\s+" + SO + r"\s+IoU\s+" + SO + r"\s+mIoU\s+" + SO, log)
    return [tuple(float(x) for x in d) for d in det], [tuple(float(x) for x in s) for s in seg]


def doc_csv(duong_dan):
    """Đọc results.csv và tách hai loại dòng: dòng phát hiện (có mAP50) và dòng phân đoạn (chỉ có loss)."""
    b = pd.read_csv(duong_dan)
    b.columns = [c.strip() for c in b.columns]
    co_det = b["metrics/mAP50(B)"].notna()
    return b[co_det].reset_index(drop=True), b[~co_det].reset_index(drop=True)


# ------------------------------------------------------------------ bảng tổng hợp
def bang_tong_hop(ds):
    """Bảng số liệu kiểm định (tập val) của từng thí nghiệm: lần kiểm định cuối trong log, cùng epoch tốt nhất theo mAP50."""
    dong = []
    for ten, f in ds.items():
        det, seg = doc_log(f["log"])
        d_csv, _ = doc_csv(f["csv"])
        r = {"Mã": ten, "Mô tả": MO_TA.get(ten, ""), "Số epoch": len(d_csv)}
        if det:
            r.update({"P": det[-1][2], "R": det[-1][3], "mAP50": det[-1][4], "mAP50-95": det[-1][5]})
        if seg:
            r.update({"pixacc": seg[-1][0], "IoU": seg[-1][2], "mIoU": seg[-1][3]})
        if len(d_csv):
            i = int(d_csv["metrics/mAP50(B)"].idxmax())
            r["Epoch mAP50 cao nhất"] = int(d_csv.loc[i, "epoch"])
            r["mAP50 cao nhất (csv)"] = float(d_csv.loc[i, "metrics/mAP50(B)"])
        dong.append(r)
    bang = pd.DataFrame(dong)
    cot = ["Mã", "Mô tả", "Số epoch"] + [c for c in CHI_SO if c in bang.columns] + \
          [c for c in ("Epoch mAP50 cao nhất", "mAP50 cao nhất (csv)") if c in bang.columns]
    return bang[cot] if len(bang) else bang


def so_sanh_cap(bang):
    """Chênh lệch giữa các cặp thí nghiệm (tuyệt đối và theo phần trăm). Số dương nghĩa là mô hình thứ nhất cao hơn."""
    b = bang.set_index("Mã")
    dong = []
    for a, c, yeu_to in CAC_CAP:
        if a not in b.index or c not in b.index:
            continue
        r = {"Cặp": f"{a} so với {c}", "Yếu tố khác nhau": yeu_to}
        for m in CHI_SO:
            if m in b.columns and pd.notna(b.loc[a, m]) and pd.notna(b.loc[c, m]):
                r[f"Δ{m}"] = round(b.loc[a, m] - b.loc[c, m], 4)
                r[f"Δ{m} (%)"] = round(100 * (b.loc[a, m] - b.loc[c, m]) / b.loc[c, m], 2) if b.loc[c, m] else None
        dong.append(r)
    return pd.DataFrame(dong)


def dinh_dang(bang, so_le=4, dau_phay=True):
    """Bảng markdown (hoặc văn bản) để dán vào báo cáo; mặc định dùng dấu phẩy thập phân kiểu Việt Nam."""
    b = bang.copy()
    for c in b.columns:
        if pd.api.types.is_float_dtype(b[c]):
            b[c] = b[c].map(lambda v: "" if pd.isna(v) else (f"{v:.{so_le}f}".replace(".", ",") if dau_phay else f"{v:.{so_le}f}"))
    cot = list(b.columns)
    dong = ["| " + " | ".join(cot) + " |", "|" + "|".join(["---"] * len(cot)) + "|"]
    for _, r in b.iterrows():
        dong.append("| " + " | ".join(str(r[c]) for c in cot) + " |")
    return "\n".join(dong)


def luu(bang, cap, thu_muc):
    """Lưu bảng tổng hợp và bảng so sánh cặp ra CSV, JSON và markdown."""
    os.makedirs(thu_muc, exist_ok=True)
    bang.to_csv(f"{thu_muc}/tong_hop.csv", index=False, encoding="utf-8-sig")
    cap.to_csv(f"{thu_muc}/so_sanh_cap.csv", index=False, encoding="utf-8-sig")
    with open(f"{thu_muc}/tong_hop.json", "w", encoding="utf-8") as f:
        json.dump(json.loads(bang.to_json(orient="records", force_ascii=False)), f, ensure_ascii=False, indent=2)
    with open(f"{thu_muc}/tong_hop.md", "w", encoding="utf-8") as f:
        f.write("## Số liệu kiểm định\n\n" + dinh_dang(bang) + "\n\n## So sánh theo cặp\n\n" + dinh_dang(cap) + "\n")
    print("Đã lưu vào", thu_muc)


# ------------------------------------------------------------------ đường cong huấn luyện
def _mau(ten):
    return {"E0": "#4C78A8", "E_main": "#2E7D32", "E1": "#F58518", "E2": "#C0392B", "E3": "#7B3F98"}.get(ten, "#555")


def ve_duong_cong(ds, thu_muc=None, hien=True):
    """Vẽ ba hình: loss huấn luyện phát hiện, chỉ số phát hiện theo epoch, chỉ số phân đoạn theo epoch. Trả về danh sách file."""
    import matplotlib
    if not hien:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = []
    if thu_muc:
        os.makedirs(thu_muc, exist_ok=True)

    # 1. loss huấn luyện của nhiệm vụ phát hiện
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for ten, f in ds.items():
        d, _ = doc_csv(f["csv"])
        for a, c, t in zip(ax, ("train/box_loss", "train/cls_loss", "train/dfl_loss"), ("box_loss", "cls_loss", "dfl_loss")):
            if c in d.columns:
                a.plot(d["epoch"], d[c], label=ten, color=_mau(ten), lw=1.6)
                a.set_title(f"Loss huấn luyện: {t}"); a.set_xlabel("epoch"); a.grid(alpha=.3)
    ax[0].legend()
    fig.tight_layout(); out.append(_xong(fig, thu_muc, "cong_loss_phat_hien.png", hien))

    # 2. chỉ số phát hiện trên tập val theo epoch (lấy từ results.csv)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for ten, f in ds.items():
        d, _ = doc_csv(f["csv"])
        for a, c, t in zip(ax, ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)"), ("mAP50", "mAP50-95", "Precision")):
            if c in d.columns:
                a.plot(d["epoch"], d[c], label=ten, color=_mau(ten), lw=1.6)
                a.set_title(f"{t} trên tập val"); a.set_xlabel("epoch"); a.grid(alpha=.3)
    ax[0].legend()
    fig.tight_layout(); out.append(_xong(fig, thu_muc, "cong_chi_so_phat_hien.png", hien))

    # 3. chỉ số phân đoạn theo epoch (lấy từ log vì results.csv không chứa)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    co = False
    for ten, f in ds.items():
        _, seg = doc_log(f["log"])
        if not seg:
            continue
        co = True
        # lần kiểm định cuối (với best.pt) không thuộc về epoch nào nên bỏ khỏi đường cong nếu số lần > số epoch
        d, _ = doc_csv(f["csv"])
        n = min(len(seg), len(d)) if len(d) else len(seg)
        for a, i, t in zip(ax, (0, 2, 3), ("pixacc", "IoU", "mIoU")):
            a.plot(range(n), [s[i] for s in seg[:n]], label=ten, color=_mau(ten), lw=1.6)
            a.set_title(f"{t} trên tập val"); a.set_xlabel("epoch"); a.grid(alpha=.3)
    if co:
        ax[0].legend()
        fig.tight_layout(); out.append(_xong(fig, thu_muc, "cong_chi_so_phan_doan.png", hien))
    else:
        plt.close(fig)
        print("Chưa có chỉ số phân đoạn trong log nên bỏ qua hình thứ ba.")
    return out


def _xong(fig, thu_muc, ten, hien):
    import matplotlib.pyplot as plt
    p = None
    if thu_muc:
        p = os.path.join(thu_muc, ten)
        fig.savefig(p, dpi=180)
    if hien:
        plt.show()
    else:
        plt.close(fig)
    return p


def bieu_do_so_sanh(bang, thu_muc=None, hien=True):
    """Biểu đồ cột so sánh các chỉ số chính giữa các thí nghiệm (mAP50, mAP50-95, IoU, mIoU)."""
    import matplotlib
    if not hien:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cot = [c for c in ("mAP50", "mAP50-95", "IoU", "mIoU") if c in bang.columns]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    w = 0.8 / max(len(bang), 1)
    for i, (_, r) in enumerate(bang.iterrows()):
        ax.bar([j + i * w for j in range(len(cot))], [r[c] for c in cot], w, label=r["Mã"], color=_mau(r["Mã"]), edgecolor="#555")
    ax.set_xticks([j + 0.4 - w / 2 for j in range(len(cot))]); ax.set_xticklabels(cot)
    ax.set_ylim(0, 1); ax.set_ylabel("Giá trị"); ax.legend(ncol=len(bang)); ax.grid(axis="y", alpha=.3)
    ax.set_title("So sánh chỉ số trên tập kiểm định")
    fig.tight_layout()
    return _xong(fig, thu_muc, "so_sanh_chi_so.png", hien)
