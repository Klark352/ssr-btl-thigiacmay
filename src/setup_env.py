"""Cài đặt môi trường YOLO-SSR trên Kaggle: tải repo gốc rồi áp dụng các bản vá cần thiết.

Mọi bản vá đều an toàn khi chạy lặp lại (đã vá rồi thì bỏ qua) và in rõ kết quả bằng tiếng Việt.
Dùng:  import setup_env; setup_env.cai_dat()
"""
import os
import re
import subprocess

REPO_GOC = "https://github.com/Mr-lxd/YOLO-SSR"
REPO = "/kaggle/working/YOLO-SSR"
# Các tên lớp lấy từ file module.py mà repo gốc không đưa lên GitHub (bị .gitignore chặn)
TEN_THIEU = ["DySample", "CARAFE", "SPDConv", "CSPOmniKernel"]


def _doc(duong_dan):
    with open(duong_dan, encoding="utf-8") as f:
        return f.read()


def _ghi(duong_dan, noi_dung):
    with open(duong_dan, "w", encoding="utf-8") as f:
        f.write(noi_dung)


def tai_repo(repo=REPO):
    """Tải repo gốc nếu chưa có."""
    if os.path.isdir(os.path.join(repo, "ultralytics")):
        return "đã có sẵn"
    subprocess.run(["git", "clone", "--depth", "1", REPO_GOC, repo], check=True)
    return "đã tải mới"


def va_init_modules(repo):
    """Bản vá 1: bỏ import từ file module.py không tồn tại và các tên lớp tương ứng trong __all__."""
    p = os.path.join(repo, "ultralytics/nn/modules/__init__.py")
    if not os.path.exists(p):
        return "không thấy file"
    goc = _doc(p)
    s = re.sub(r"^from \.module import \(.*?\)[ \t]*\n", "", goc, flags=re.S | re.M)
    s = re.sub(r"^from \.module import [^\n(]*\n", "", s, flags=re.M)
    for ten in TEN_THIEU:
        s = re.sub(r",\s*(['\"])%s\1" % ten, "", s)
    if s == goc:
        return "đã vá từ trước (hoặc không có gì để vá)"
    _ghi(p, s)
    return "đã vá"


def _them_weights_only(s):
    """Thêm weights_only=False vào mọi lời gọi torch.load( ... ) chưa có tham số này."""
    ra, i, dem = [], 0, 0
    while True:
        j = s.find("torch.load(", i)
        if j < 0:
            ra.append(s[i:])
            break
        k = j + len("torch.load(")
        sau, m = 1, k
        while m < len(s) and sau:
            sau += (s[m] == "(") - (s[m] == ")")
            m += 1
        trong = s[k:m - 1]
        ra.append(s[i:k])
        if "weights_only" in trong or not trong.strip():
            ra.append(trong)
        else:
            ra.append(trong.rstrip().rstrip(",") + ", weights_only=False")
            dem += 1
        ra.append(")")
        i = m
    return "".join(ra), dem


def va_torch_load(repo):
    """Bản vá 2: PyTorch mới mặc định chặn nạp checkpoint đầy đủ, nên đặt weights_only=False."""
    ket_qua = []
    for tep in ["ultralytics/nn/tasks.py", "ultralytics/yolo/utils/torch_utils.py"]:
        p = os.path.join(repo, tep)
        if not os.path.exists(p):
            ket_qua.append(f"{os.path.basename(tep)}: không thấy file")
            continue
        s, dem = _them_weights_only(_doc(p))
        if dem:
            _ghi(p, s)
        ket_qua.append(f"{os.path.basename(tep)}: {'vá ' + str(dem) + ' chỗ' if dem else 'đã vá từ trước'}")
    return "; ".join(ket_qua)


def va_tensorboard(repo):
    """Bản vá 3: callback TensorBoard của repo gọi .items() trên một danh sách, gây lỗi."""
    p = os.path.join(repo, "ultralytics/yolo/utils/callbacks/tensorboard.py")
    if not os.path.exists(p):
        return "không thấy file"
    s = _doc(p)
    if "hasattr(scalars, 'items')" in s or 'hasattr(scalars, "items")' in s:
        return "đã vá từ trước"
    moi, dem = re.subn(r"(def _log_scalars\([^)]*\):\n)",
                       r"\1    if not hasattr(scalars, 'items'):\n        return\n", s, count=1)
    if not dem:
        return "không thấy hàm _log_scalars"
    _ghi(p, moi)
    return "đã vá"


def va_albumentations(repo):
    """Bản vá 4: tắt dòng in danh sách phép biến đổi Albumentations (làm nhiễu log)."""
    p = os.path.join(repo, "ultralytics/yolo/data/augment.py")
    if not os.path.exists(p):
        return "không thấy file"
    s = _doc(p)
    moi, dem = re.subn(r"^(\s*)(?:print|LOGGER\.info)\(.*lbumentations.*\)[ \t]*$", r"\1pass", s, flags=re.M)
    if not dem:
        return "đã vá từ trước (hoặc không có dòng in)"
    _ghi(p, moi)
    return f"đã vá {dem} dòng"


def va_numpy(repo):
    """Bản vá 5: numpy 2 đã bỏ np.float, đổi sang float (nếu repo còn dùng)."""
    p = os.path.join(repo, "ultralytics/nn/autobackend.py")
    if not os.path.exists(p):
        return "không thấy file"
    s = _doc(p)
    if "astype(np.float)" not in s:
        return "không cần vá"
    _ghi(p, s.replace("astype(np.float)", "astype(float)"))
    return "đã vá"


def kiem_tra_tep(repo):
    """Kiểm tra các file mà bước suy luận và huấn luyện cần."""
    can = ["module_ours.py", "JetsonNano_Nav_Visualization.py"]
    thieu = []
    for ten in can:
        if not any(ten in files for _, _, files in os.walk(repo)):
            thieu.append(ten)
    return thieu


def cai_dat(repo=REPO):
    """Tải repo, vá lỗi tương thích, in báo cáo. Trả về đường dẫn repo."""
    print("== CÀI ĐẶT MÔI TRƯỜNG YOLO-SSR ==")
    print("Repo gốc          :", tai_repo(repo))
    print("Vá __init__.py    :", va_init_modules(repo))
    print("Vá torch.load     :", va_torch_load(repo))
    print("Vá TensorBoard    :", va_tensorboard(repo))
    print("Vá Albumentations :", va_albumentations(repo))
    print("Vá numpy          :", va_numpy(repo))
    thieu = kiem_tra_tep(repo)
    if thieu:
        print("CẢNH BÁO: không thấy file", ", ".join(thieu), "trong repo, kiểm tra lại bước tải repo.")
    else:
        print("Các file cần thiết đều có trong repo.")
    return repo


if __name__ == "__main__":
    cai_dat()
