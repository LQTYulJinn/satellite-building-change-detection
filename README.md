# Phát hiện thay đổi công trình trên ảnh vệ tinh

Project dùng **LEVIR-CD** để tìm vùng công trình thay đổi giữa ảnh trước và sau. Có ba cách so sánh:

1. Baseline: độ chênh lệch RGB giữa hai ảnh, ngưỡng chọn trên tập validation.
2. Early fusion U-Net: ghép hai ảnh RGB thành đầu vào 6 kênh.
3. Siamese U-Net: hai ảnh đi qua cùng encoder, ghép đặc trưng ở decoder.

Đầu ra là mask nhị phân. Báo cáo IoU, F1, precision và recall của lớp **thay đổi**, cộng hình trước/sau/nhãn/dự đoán/lỗi.

Đọc [PROJECT_REPORT.md](PROJECT_REPORT.md) để xem câu hỏi nghiên cứu, lý do chọn dữ liệu/mô hình/loss, công thức và ý nghĩa các độ đo, thiết kế thí nghiệm, giới hạn và cách mô tả trung thực trên CV. **Trạng thái:** pipeline đã viết; chưa có dữ liệu và checkpoint huấn luyện trong workspace, nên chưa có kết quả thật.

## 1. Chuẩn bị

Cần Python 3.10+ và nên có GPU NVIDIA để huấn luyện nhanh. Trên Windows có `uv`, cài trong PowerShell:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv venv .venv --python 3.12
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

Sau khi kích hoạt `.venv`, các lệnh `python` bên dưới sẽ dùng đúng môi trường này. Nếu không dùng `uv`, hãy tạo virtual environment bằng Python 3.10+ và cài `pip install -r requirements.txt`.

Kiểm tra GPU bằng `python -c "import torch; print(torch.cuda.is_available())"`. Nếu in `False`, pipeline vẫn chạy trên CPU nhưng huấn luyện toàn bộ LEVIR-CD sẽ chậm; để hoàn tất thí nghiệm chính, nên dùng máy có GPU hoặc chuyển cùng mã và dữ liệu sang một môi trường notebook có GPU.

Kiểm tra mã trước khi có dữ liệu thật:

```powershell
python -m unittest discover -s tests -v
python smoke_test.py
```

`smoke_test.py` chạy toàn tuyến trên ảnh giả rất nhỏ và xóa chúng khi hoàn tất. Điểm từ ảnh giả **không phải** kết quả nghiên cứu.

Tải LEVIR-CD từ [trang chính thức](https://justchenhao.github.io/LEVIR/) bằng nút **Google Drive** ([thư mục tải trực tiếp](https://drive.google.com/drive/folders/1dLuzldMRmbBNKPpUkX8Z53hi6NHLrWim)) hoặc Baidu Drive (mã `l7iv`). Dữ liệu chỉ được tác giả cho phép sử dụng cho mục đích học thuật. Trong thư mục tải, chọn bản LEVIR-CD có `train/val/test` và ảnh gốc 1024×1024; không dùng LEVIR-CD+ cho thí nghiệm này. Giải nén và đặt dữ liệu theo cấu trúc:

```text
data/LEVIR-CD/
  train/A/*.png
  train/B/*.png
  train/label/*.png
  val/A/*.png
  val/B/*.png
  val/label/*.png
  test/A/*.png
  test/B/*.png
  test/label/*.png
```

Tên file ở A, B và label phải giống nhau trong từng split. Nếu bản tải về có tên thư mục khác, đổi tên hoặc truyền `--data` tới thư mục đã chuẩn hóa. **Chia train/val/test theo ảnh gốc trước khi cắt patch**; không chia ngẫu nhiên các patch từ cùng một ảnh sang nhiều split. Script đọc split riêng và cắt patch 256×256 trong bộ nhớ. Mask nhận giá trị 0/1 hoặc 0/255.

Sau khi giải nén, mở `data/LEVIR-CD/train/A` và xác nhận có file `.png`; làm tương tự với `B` và `label` ở cả ba split. Nếu file tải về có thêm một tầng thư mục (ví dụ `data/LEVIR-CD/LEVIR-CD/train`), chuyển **thư mục chứa trực tiếp `train`, `val`, `test`** thành `data/LEVIR-CD` hoặc truyền đúng đường dẫn đó qua `--data`. Chạy `python audit_data.py --data data/LEVIR-CD`; chỉ bắt đầu huấn luyện khi lệnh này thành công và báo số ảnh, tỷ lệ pixel thay đổi cho từng split. Cấu trúc split này cũng được xác nhận trong [mã chính thức STANet của tác giả](https://github.com/justchenhao/STANet#prepare-datasets).

## 2. Chạy thí nghiệm

```powershell
python audit_data.py --data data/LEVIR-CD
python baseline.py --data data/LEVIR-CD
python train.py --data data/LEVIR-CD --architecture early --run runs/early --epochs 30
python evaluate.py --data data/LEVIR-CD --checkpoint runs/early/best.pt --split test
```

Nếu muốn so sánh hai mô hình:

```powershell
python train.py --data data/LEVIR-CD --architecture siamese --run runs/siamese --epochs 30
python evaluate.py --data data/LEVIR-CD --checkpoint runs/siamese/best.pt --split test
python summarize.py
```

`audit_data.py` kiểm tra các bộ ba ảnh và tỉ lệ pixel thay đổi trước khi train. `train.py` tự chọn GPU nếu có, lưu mô hình tốt nhất theo F1 trên validation vào `best.pt`, ghi lịch sử vào `history.csv` và thời gian/số tham số vào `training_summary.json`. `evaluate.py` chọn ngưỡng trên validation rồi ghi `metrics.json`, `per_tile.csv` và hình PNG vào `runs/<tên run>/test/`. `summarize.py` tạo bảng `runs/results.md` từ các số đã đo. Trên máy ít VRAM, giảm `--batch-size` xuống 2 hoặc `--base` xuống 16. Dùng cùng `--base`, số epoch, seed và split khi so sánh hai mô hình.

## 3. Trình bày kết quả

| Phương pháp | IoU test | F1 test | Precision | Recall |
|---|---:|---:|---:|---:|
| RGB difference | điền sau khi chạy | | | |
| Early fusion U-Net | | | | |
| Siamese U-Net | | | | |

Xem hình đúng, false positive và false negative từ thư mục `examples` để thảo luận. Trong hình lỗi: xanh lá = dự đoán đúng vùng thay đổi, đỏ = báo thay đổi sai, xanh dương = bỏ sót. Không chọn ngưỡng hoặc siêu tham số dựa trên test set. Bảng trong phần này là mẫu; chỉ thay số sau khi chạy thực tế.

## Giới hạn hiện tại

Pipeline giả định ảnh A/B đã căn chỉnh, có cùng kích thước, định dạng PNG, và chỉ đánh giá các patch hoàn chỉnh. Mã chưa tải dữ liệu tự động vì bản LEVIR-CD được cung cấp qua liên kết của tác giả và có dung lượng lớn. Kết quả định lượng sẽ có sau khi đặt dữ liệu vào `data/` và chạy huấn luyện.
