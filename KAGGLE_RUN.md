# Chạy thí nghiệm thật trên Kaggle GPU

Tài liệu này dành cho việc tạo kết quả trên **LEVIR-CD thật**. Không dùng kết quả của `smoke_test.py` trong CV.

## Gửi qua Kaggle API

Thư mục [`kaggle_kernel/`](kaggle_kernel/) chứa script và metadata để gửi một phiên chạy GPU bằng Kaggle CLI. Sau khi tự đăng nhập với `kaggle auth login`, chạy từ thư mục gốc project:

```powershell
kaggle kernels push -p kaggle_kernel --accelerator NvidiaTeslaT4
kaggle kernels status yuljinn/levir-cd-u-net-building-change-detection
kaggle kernels logs yuljinn/levir-cd-u-net-building-change-detection
```

Kaggle kernel ở chế độ private, gắn bản sao LEVIR-CD trên Kaggle và chạy đúng commit GitHub được ghi trong `kaggle_kernel/run.py`. Chỉ chia sẻ notebook hoặc điểm số sau khi đã kiểm tra đầu ra thật.

## Chuẩn bị notebook

1. Tạo Kaggle Notebook, bật **GPU** trong phần Accelerator và bật Internet cho bước tải mã.
2. Thêm dataset [LEVIR-CD trên Kaggle](https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd) vào notebook. Đây là bản sao do bên thứ ba đăng; đối chiếu cấu trúc với [nguồn chính thức của tác giả](https://justchenhao.github.io/LEVIR/). Điều khoản của tác giả chỉ cho phép dùng học thuật; thông tin giấy phép trên bản sao không thay thế điều khoản đó.
3. Chạy các cell bên dưới theo thứ tự. Đầu ra được lưu trong `/kaggle/working/satellite-building-change-detection/runs`.

```python
!git clone https://github.com/LQTYulJinn/satellite-building-change-detection.git
%cd /kaggle/working/satellite-building-change-detection
```

```python
from pathlib import Path
import torch

matches = [p.parent for p in Path('/kaggle/input').rglob('train')
           if p.is_dir() and (p / 'A').is_dir()
           and (p.parent / 'val' / 'A').is_dir()
           and (p.parent / 'test' / 'A').is_dir()]
assert len(matches) == 1, f'Không tìm thấy đúng một bộ LEVIR-CD: {matches}'
DATA = matches[0]
assert torch.cuda.is_available(), 'Notebook chưa bật GPU'
print('GPU:', torch.cuda.get_device_name(0))
print('Data:', DATA)
```

```python
import subprocess, sys

def run(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True)

run('audit_data.py', '--data', DATA)
run('baseline.py', '--data', DATA)
```

```python
run('train.py', '--data', DATA, '--architecture', 'early', '--run', 'runs/early',
    '--epochs', '30', '--batch-size', '8', '--base', '32', '--workers', '4', '--seed', '42')
run('evaluate.py', '--data', DATA, '--checkpoint', 'runs/early/best.pt', '--split', 'test')
```

```python
run('train.py', '--data', DATA, '--architecture', 'siamese', '--run', 'runs/siamese',
    '--epochs', '30', '--batch-size', '8', '--base', '32', '--workers', '4', '--seed', '42')
run('evaluate.py', '--data', DATA, '--checkpoint', 'runs/siamese/best.pt', '--split', 'test')
run('summarize.py')
print(Path('runs/results.md').read_text())
```

Nếu notebook hết phiên giữa chừng, lưu phiên bản sau mỗi mô hình để giữ `best.pt`. `train.py` hiện chưa hỗ trợ resume giữa epoch; không ghi nhận mô hình chưa chạy đủ kế hoạch là kết quả cuối. Nếu hết VRAM, giảm `--batch-size` xuống 4 hoặc 2 cho **cả hai** mô hình, ghi lại cấu hình trong báo cáo. Kaggle có thể giới hạn thời lượng GPU; kiểm tra quota trước khi bắt đầu.

## Mang kết quả về repo

Tải các file `runs/baseline.json`, `runs/early/test/metrics.json`, `runs/siamese/test/metrics.json`, `training_summary.json` và vài hình `examples/*.png` từ đầu ra notebook. Giữ checkpoint lớn và dữ liệu ở ngoài Git thường; chỉ đưa số liệu/hình thật, cấu hình thí nghiệm và nguồn dữ liệu vào README sau khi kiểm tra. Nếu muốn phát hành checkpoint, dùng GitHub Release hoặc kho model phù hợp và ghi rõ điều khoản LEVIR-CD.
