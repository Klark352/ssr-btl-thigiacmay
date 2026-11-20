# Hướng dẫn giai đoạn 1 (giờ 0:00 đến 2:00 của ngày 1)

Mục tiêu: có repo GitHub sạch, môi trường Kaggle tái lập được, và 3 lần chạy thử 1 epoch thành công
(mô hình chính, baseline, bản bỏ module). Làm đúng thứ tự dưới đây.

## A. VS Code và GitHub (khoảng 20 phút)

1. Giải nén thư mục `ssr-btl-thigiacmay` vào `E:\btl_thigiacmay\`. Mở thư mục đó bằng VS Code (File > Open Folder).
   *Lý do:* VS Code là nơi bạn sửa mã và cấu hình, mọi thay đổi đi qua git nên có lịch sử và chạy lại được.
2. Vào github.com, tạo repo mới tên `ssr-btl-thigiacmay`, chế độ **Public**, **không** tick "Add README".
   *Lý do:* Public để notebook Kaggle tải về được mà không cần mật khẩu, và giảng viên xem được; không tick README để lần push đầu không bị xung đột.
3. Mở Terminal trong VS Code (PowerShell) và chạy:
   ```
   git init -b main
   git add .
   git commit -m "Khung dự án BTL YOLO-SSR"
   git remote add origin https://github.com/<TEN_GITHUB>/ssr-btl-thigiacmay.git
   git push -u origin main
   ```
   Lần đầu VS Code sẽ mở trình duyệt để đăng nhập GitHub. Nếu `git` chưa cài, tải tại git-scm.com.
   *Lý do:* `.gitignore` đã chặn file `.pt`, `.zip`, dữ liệu và thư mục `YOLO-SSR/` để repo chỉ chứa mã của bạn, nhẹ và đúng giấy phép.

## B. Kaggle: tạo và cấu hình notebook (khoảng 15 phút)

1. kaggle.com > Create > New Notebook > File > Import Notebook > chọn `notebooks/01_chuan_bi_va_chay_thu.ipynb`.
2. Panel Settings bên phải: Accelerator = GPU P100 (hoặc T4 x1), Internet = On.
   *Lý do:* mã huấn luyện dùng `device=[0]` tức một GPU; Internet cần để `git clone` repo của bạn và repo gốc.
3. Add Input > tìm `seedlingstemrow` của `xxdxdxd` > Add.
4. Ở cell đầu sửa `GITHUB_USER` thành tên tài khoản GitHub của bạn.

## C. Chạy từng cell của notebook 01

| Cell | Làm gì | Thành công khi |
|---|---|---|
| Bước 1 | Tải mã nguồn của bạn từ GitHub | In ra `['setup_env.py', 'ssr_lib.py']` |
| Bước 2 | Tải repo gốc, vá 5 chỗ tương thích | Mỗi dòng báo "đã vá" hoặc "đã vá từ trước", cuối cùng "Các file cần thiết đều có" |
| Bước 3 | Thu nhỏ ảnh (cạnh dài tối đa 1280), tạo `ssr-multi.yaml` | In số ảnh train, val, test và tên lớp |
| Bước 4 | Ghi phiên bản thư viện và commit repo gốc | In Python, torch, GPU |
| Bước 5 | Liệt kê yaml, README, trọng số của repo | **Gửi kết quả cho mình** để chốt baseline |
| Bước 6 | Chạy thử 1 epoch với E_main | In `HOÀN THÀNH` và bảng số liệu epoch cuối |
| Bước 7 | Chạy thử 1 epoch với baseline E0 | Làm sau khi điền `cfg` của E0 |
| Bước 8 | Chạy thử 1 epoch với bản bỏ module E2 | Làm sau khi điền `cfg` của E2 |

**Bước 3 là chỗ cần kiểm tra bằng mắt:** số ảnh mỗi tập phải khớp với dữ liệu gốc, và tên lớp phải đúng.
Nếu thấy dòng "CẢNH BÁO: không thấy tên lớp", sửa lại `names` trong `/kaggle/working/ssr-multi.yaml`.
Bước 3 lần đầu mất khá lâu vì phải thu nhỏ toàn bộ ảnh. Sau khi xong, nên lưu thư mục `/kaggle/working/ssr` thành một Kaggle Dataset
(Output > New Dataset) để các notebook sau chỉ cần chép về thay vì thu nhỏ lại; khi đó gọi `L.chuan_bi_du_lieu("/kaggle/input/<ten-dataset>")`.

## D. Sau khi có kết quả Bước 5

1. Gửi mình kết quả: mình chốt baseline và chỉ ra module nên bỏ ở E2, kèm file yaml.
2. Bạn điền `cfg` của E0 và E2 vào `configs/experiments.json` (VS Code), rồi:
   ```
   git add configs/experiments.json
   git commit -m "Điền cấu hình E0 và E2"
   git push
   ```
3. Trong notebook Kaggle chạy lại Bước 1 (tự `git pull`), rồi chạy Bước 7 và Bước 8.

## E. Gặp lỗi thì xem

| Hiện tượng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `RuntimeError ... deterministic implementation` | `deterministic=True` chạm phép toán GPU không xác định | Đặt `"deterministic": false` trong `configs/experiments.json`, ghi lại điều này vào báo cáo |
| `Không tìm thấy thư mục ảnh train hoặc val` | Cấu trúc dữ liệu khác dự kiến | Gửi mình kết quả `os.walk` hai cấp của `/kaggle/working/ssr` |
| `Weights only load failed` | Bản vá `torch.load` không áp dụng được | Gửi nguyên văn lỗi và kết quả Bước 2 |
| `CUDA out of memory` | Batch quá lớn | Thêm `"batch": 8` vào `chung`, chạy lại **mọi** thí nghiệm với cùng giá trị |
| Mất `/kaggle/working` sau khi phiên khởi động lại | Kaggle xóa khi phiên kết thúc | Chạy các thí nghiệm dài bằng *Save & Run All* và tải `.zip` về |
