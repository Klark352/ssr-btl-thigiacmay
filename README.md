# BTL Thị giác máy: YOLO-SSR (phát hiện thân cây và phân đoạn hàng cây)

Bài tập lớn môn Thị giác máy, Học viện Kỹ thuật Quân sự. Tái lập và đánh giá mô hình
[YOLO-SSR](https://github.com/Mr-lxd/YOLO-SSR) (giấy phép AGPL-3.0) trên bộ dữ liệu SSR.
Repo này **không** chứa mã của mô hình gốc: chỉ chứa mã cài đặt, cấu hình thí nghiệm, notebook chạy trên Kaggle
và kết quả tóm tắt của riêng tôi.

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `src/setup_env.py` | Tải repo gốc và vá các lỗi tương thích (PyTorch 2.x, numpy 2, callback ghi log) |
| `src/ssr_lib.py` | Chuẩn bị dữ liệu, huấn luyện, ghi log môi trường, tóm tắt và nén kết quả |
| `configs/experiments.json` | Định nghĩa các thí nghiệm E0, E_main, E1, E2, E3 |
| `notebooks/01_chuan_bi_va_chay_thu.ipynb` | Chuẩn bị môi trường, chốt baseline, chạy thử 1 epoch |
| `notebooks/02_chay_thi_nghiem.ipynb` | Chạy một thí nghiệm theo mã |
| `ket_qua/` | File CSV tổng hợp số liệu (thêm sau khi chạy xong) |

## Cách tái lập trên Kaggle

1. Tạo notebook mới, nhập (Import) `notebooks/01_chuan_bi_va_chay_thu.ipynb`.
2. Cài đặt: Accelerator = GPU, Internet = On, thêm dữ liệu `xxdxdxd/seedlingstemrow`.
3. Sửa `GITHUB_USER` ở cell đầu, chạy lần lượt các cell. Kết thúc bằng 1 epoch chạy thử thành công.
4. Với từng thí nghiệm: nhập `02_chay_thi_nghiem.ipynb`, đặt `MA_THI_NGHIEM`, chọn *Save & Run All (Commit)*.

Mọi lần chạy dùng `seed=0`, `deterministic=True`, cùng bộ dữ liệu; chỉ đổi một yếu tố giữa các thí nghiệm.
Phiên bản thư viện và commit của repo gốc được ghi vào `logs/moi_truong.txt` ở mỗi lần chạy.

## Thí nghiệm

| Mã | Nội dung | Kết quả |
|---|---|---|
| E0 | Baseline | (điền sau) |
| E_main | YOLO-SSR đầy đủ | (điền sau) |
| E1 | Đổi hyperparameter | (điền sau) |
| E2 | Bỏ một module | (điền sau) |

## Ghi công và giấy phép

Mô hình, mã mô hình và ý tưởng thuộc nhóm tác giả YOLO-SSR (Mr-lxd) theo AGPL-3.0, xây trên Ultralytics 8.0.105.
Bộ dữ liệu SSR thuộc tác giả `xxdxdxd` trên Kaggle.
