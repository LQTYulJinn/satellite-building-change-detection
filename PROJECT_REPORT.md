# Building Change Detection with LEVIR-CD

## 1. Câu hỏi nghiên cứu

Từ hai ảnh RGB đã căn chỉnh của cùng một khu vực ở hai thời điểm, dự đoán **pixel nào thuộc vùng công trình thay đổi**. Đây là binary semantic segmentation; nhãn `1` nghĩa là công trình xuất hiện hoặc biến mất, `0` là không có thay đổi công trình. Thay đổi ánh sáng, mùa vụ hoặc cây cối không phải nhãn đích. [Tác giả LEVIR-CD](https://github.com/justchenhao/LEVIR) mô tả 637 cặp ảnh 1024×1024, độ phân giải khoảng 0,5 m/pixel từ nhiều khu vực ở Texas và nhãn công trình thay đổi.

**Giả thuyết:** mô hình học từ hai ảnh sẽ phân biệt thay đổi công trình tốt hơn ngưỡng chênh lệch RGB; Siamese U-Net có thể giảm báo sai do biến đổi bề ngoài so với early fusion U-Net. Phần thứ hai là giả thuyết cần kiểm chứng, không phải kết luận sẵn.

## 2. Vì sao chọn dữ liệu và phương pháp

| Thành phần | Lựa chọn | Lý do | Điều cần kiểm chứng |
|---|---|---|---|
| Dữ liệu | LEVIR-CD | Cặp ảnh trước/sau có nhãn pixel, đúng bài toán công trình thay đổi | Phân bố địa lý hẹp; không suy rộng ngay sang nơi khác |
| Baseline | Chênh lệch RGB trung bình + ngưỡng | Mốc đơn giản, dễ giải thích; cho thấy lợi ích thực của deep learning | Dễ nhầm bóng, màu và mùa vụ |
| Early fusion | Nối A và B thành 6 kênh, U-Net | Mô hình học so sánh trực tiếp ở mọi tầng; thiết lập chuẩn, chi phí vừa | Không buộc đặc trưng hai thời điểm có cùng cách biểu diễn |
| Siamese | Encoder chia sẻ trọng số cho A/B, ghép đặc trưng ở decoder | So sánh hai thời điểm qua cùng bộ trích đặc trưng | Có thể tốn thời gian hơn, chưa chắc điểm cao hơn |
| Loss | BCEWithLogits + soft Dice | BCE học dự đoán pixel; Dice khuyến khích chồng lấp vùng thay đổi vốn thưa | So với BCE đơn thuần ở một thí nghiệm ablation nếu có đủ thời gian |
| Augmentation | Lật và quay 90° đồng bộ A/B/mask | Tăng đa dạng hướng công trình mà không làm lệch cặp ảnh | Không biến đổi A/B riêng vì phá căn chỉnh |

U-Net có skip connection để giữ chi tiết biên trong khi encoder học ngữ cảnh rộng. `base=32` là cấu hình mặc định; có thể giảm xuống 16 nếu thiếu VRAM. Cả hai mô hình được huấn luyện cùng số epoch, seed, patch size và optimizer để so sánh công bằng. Số tham số của hai kiến trúc không hoàn toàn bằng nhau, vì vậy chênh lệch kết quả không thể quy toàn bộ cho riêng cách kết hợp hai ảnh.

## 3. Quy trình dữ liệu và chống rò rỉ

1. Dùng split `train/val/test` đã tách ở cấp ảnh gốc; chỉ sau đó mới cắt mỗi ảnh thành patch 256×256 không chồng lấp.
2. Chạy `audit_data.py`: kiểm tra A/B/label cùng tên và kích thước, nhãn chỉ gồm 0/1 hoặc 0/255, tỷ lệ pixel thay đổi và cặp ảnh giống hệt giữa các split.
3. Chỉ áp dụng augmentation cho train. Không dùng test để chọn checkpoint, ngưỡng, kiến trúc hoặc siêu tham số.
4. Chọn checkpoint theo F1 trên validation tại ngưỡng 0,5. Sau huấn luyện, tìm ngưỡng dự đoán tốt nhất trên validation (0,10–0,90, bước 0,05), rồi **khóa ngưỡng** để chấm test.

Ảnh được chuẩn hóa về `[0,1]`; mask được chuyển thành 0/1. Pipeline giả định A và B đã đăng ký không gian. Pixel lệch do đăng ký sai có thể tạo false positive gần mép công trình.

## 4. Vì sao cần các độ đo này

Gọi TP là pixel thay đổi dự đoán đúng; FP là pixel bị báo thay đổi sai; FN là pixel thay đổi bị bỏ sót.

| Độ đo | Công thức | Trả lời câu hỏi |
|---|---|---|
| Precision | `TP / (TP + FP)` | Khi mô hình báo thay đổi, bao nhiêu phần là đúng? |
| Recall | `TP / (TP + FN)` | Bao nhiêu vùng thay đổi thật được tìm thấy? |
| F1 | `2TP / (2TP + FP + FN)` | Cân bằng precision và recall tại một ngưỡng |
| IoU | `TP / (TP + FP + FN)` | Mức chồng lấp của mask dự đoán với nhãn thật |

**IoU và F1 là chỉ số chính** vì vùng thay đổi thường chiếm ít pixel. Accuracy có thể rất cao nếu mô hình dự đoán toàn bộ là “không thay đổi”, nên không dùng làm thước đo chính. Với cùng TP/FP/FN, `F1 = 2·IoU / (1 + IoU)`; báo cả hai giúp người đọc đối chiếu chuẩn segmentation, nhưng chúng không phải hai bằng chứng độc lập. Precision và recall giúp phân biệt mô hình hay báo sai với mô hình hay bỏ sót. `evaluate.py` cộng TP/FP/FN trên toàn bộ test rồi tính điểm (micro average), không lấy trung bình F1 của từng patch vì patch không có thay đổi có thể gây diễn giải sai.

## 5. Thiết kế thí nghiệm

**Thí nghiệm chính:** chạy RGB difference, early fusion U-Net và Siamese U-Net trên cùng dữ liệu. `baseline.py` chọn ngưỡng trên validation; `evaluate.py` chọn ngưỡng mô hình trên validation. Báo bảng từ `summarize.py`.

**Ablation đề xuất sau khi có kết quả chính:** BCE so với BCE+Dice; hoặc patch 256 so với 512. Chỉ thay đổi **một** yếu tố mỗi lần, dùng cùng split/seed, rồi so sánh F1, IoU và FP/FN. Một lần chạy chưa đủ chứng minh cải thiện nhỏ là ổn định; nếu có tài nguyên, lặp 3 seed và báo trung bình ± độ lệch chuẩn.

**Phân tích lỗi:** `per_tile.csv` ghi TP/FP/FN cho từng patch. Hình trong `examples/` chọn từ các trường hợp nhiều FN, nhiều FP, F1 thấp/cao và patch không có thay đổi. Khi xem, ghi lại nguyên nhân khả dĩ như bóng, thay đổi mùa, công trình nhỏ, biên nhãn hoặc ảnh lệch. Đây là diễn giải định tính; không khẳng định nguyên nhân nếu chưa kiểm tra ảnh gốc.

## 6. Kết quả

Kết quả dưới đây là từ hai lần chạy notebook Kaggle của project, mỗi lần 30 epoch với cùng cấu hình chính. Lần 1 được khôi phục từ output notebook; lần 2 có ZIP đầy đủ và checkpoint `best.pt` đã được kiểm tra tải bằng PyTorch. Các số được lưu trong [`recovered_results/`](recovered_results/) và [`results/early_run_2/`](results/early_run_2/). Không dùng số từ bài báo của người khác.

| Phương pháp | IoU test | F1 test | Precision | Recall | Thời gian huấn luyện |
|---|---:|---:|---:|---:|---:|
| RGB difference | 0,0700 | 0,1308 | 0,1040 | 0,1760 | không huấn luyện |
| Early fusion U-Net, lần 1 | 0,5834 | 0,7369 | 0,7454 | 0,7286 | 206 phút |
| Early fusion U-Net, lần 2 | 0,5951 | 0,7461 | 0,7784 | 0,7164 | 206 phút |
| Siamese U-Net | chưa chạy | chưa chạy | chưa chạy | chưa chạy | chưa chạy |

Lần 2 chọn checkpoint tốt nhất ở epoch 29 và ngưỡng 0,30 trên validation. Test có TP = 4.898.411, FP = 1.394.414, FN = 1.938.993; điểm là micro average theo pixel. Lần 1 chọn epoch 27 và ngưỡng 0,45. Baseline RGB difference dùng ngưỡng 0,40 chọn trên validation. Dữ liệu Kaggle mirror có một số ít pixel nhãn xám trung gian; `audit_data.py` ghi nhận số lượng và `data.py` nhị phân hóa ở ngưỡng 127. Hai lần early-fusion cùng cấu hình chính nên chênh lệch điểm có thể do tính không tất định của quá trình huấn luyện, không chứng minh phương pháp mới tốt hơn. Test đã được xem sau hai lần chạy; các so sánh tiếp theo trên cùng test cần ghi là thăm dò.

Phân tích `per_tile.csv` lần 2: 1.113/2.048 tile test không có pixel thay đổi; chúng tạo 120.448/1.394.414 FP (8,6%). Ví dụ nhiều FN và FP nằm trong [`results/early_run_2/examples/`](results/early_run_2/examples/). Kế hoạch cải thiện có thứ tự và giả thuyết kiểm chứng tại [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md).

## 7. Giới hạn và đạo đức sử dụng

- LEVIR-CD tập trung vào một số khu vực Texas, nên cần dữ liệu ngoài miền để đánh giá khả năng dùng ở Việt Nam hoặc khu vực khác.
- Đây là bài toán trên ảnh **RGB độ phân giải cao**. Pipeline chưa đọc kênh đa phổ Sentinel/Landsat, chưa gắn tọa độ địa lý hoặc xuất GeoTIFF; không nên mô tả project như một hệ thống phân tích viễn thám đa phổ/địa không gian đầy đủ.
- Nhãn nhị phân không phân biệt công trình xây mới với công trình bị phá dỡ. Muốn dự đoán hướng thay đổi cần nhãn hoặc quy trình khác.
- Patch không chồng lấp làm mất ngữ cảnh ở rìa; suy luận ảnh lớn ngoài benchmark nên dùng sliding window có overlap hoặc padding.
- IoU/F1 theo pixel chưa phản ánh đầy đủ việc phát hiện từng công trình. Nếu ứng dụng cần đếm công trình, nên bổ sung đánh giá theo đối tượng.
- Tác giả dữ liệu ghi rõ ảnh và nhãn chỉ dành cho **mục đích học thuật**, không dùng thương mại; cần tuân thủ [điều khoản LEVIR-CD](https://github.com/justchenhao/LEVIR#usage).

## 8. Cách trình bày trên CV và khi phỏng vấn

Bullet CV phù hợp với kết quả đã chạy:

> Built and evaluated an early-fusion U-Net for LEVIR-CD building change detection, reaching **0.746 test F1** and **0.595 test IoU** with a saved checkpoint; compared against a **0.131 F1** RGB-difference baseline and analyzed pixel-level errors.

Không ghi đã so sánh Siamese U-Net khi thí nghiệm đó chưa chạy. Checkpoint lần 2 có trong ZIP gốc tải từ Kaggle và ở `runs/early/best.pt` trong workspace hiện tại; file này không được đưa vào Git thường.

> Developing a LEVIR-CD building change detection benchmark with paired-image U-Nets, split-leakage checks, IoU/F1 evaluation, and visual false-positive/false-negative analysis.

Ba câu nên trả lời được khi phỏng vấn:

1. **Tại sao không dùng accuracy?** Vì lớp không thay đổi áp đảo; dự đoán toàn nền vẫn có accuracy cao nhưng không tìm được công trình thay đổi.
2. **Tại sao ngưỡng chọn ở validation?** Chọn trên test sẽ đưa thông tin từ test vào quyết định mô hình, làm điểm test lạc quan.
3. **Tại sao cần baseline đơn giản?** Để đo xem mạng học được gì vượt lên trên khác biệt màu sắc; nếu mạng không hơn baseline, phải kiểm tra dữ liệu và pipeline trước khi tuyên bố thành công.
