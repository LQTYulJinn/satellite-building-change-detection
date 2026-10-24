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

Kết quả định lượng **chưa có** cho đến khi tải LEVIR-CD và chạy huấn luyện. Không điền số từ bài báo của người khác vào bảng của project. Sau khi chạy `summarize.py`, dùng `runs/results.md` làm nguồn số liệu và đưa hình thật từ `runs/<model>/test/examples/` vào README hoặc portfolio.

| Phương pháp | IoU test | F1 test | Precision | Recall | Thời gian huấn luyện |
|---|---:|---:|---:|---:|---:|
| RGB difference | chờ chạy | chờ chạy | chờ chạy | chờ chạy | không huấn luyện |
| Early fusion U-Net | chờ chạy | chờ chạy | chờ chạy | chờ chạy | chờ chạy |
| Siamese U-Net | chờ chạy | chờ chạy | chờ chạy | chờ chạy | chờ chạy |

## 7. Giới hạn và đạo đức sử dụng

- LEVIR-CD tập trung vào một số khu vực Texas, nên cần dữ liệu ngoài miền để đánh giá khả năng dùng ở Việt Nam hoặc khu vực khác.
- Nhãn nhị phân không phân biệt công trình xây mới với công trình bị phá dỡ. Muốn dự đoán hướng thay đổi cần nhãn hoặc quy trình khác.
- Patch không chồng lấp làm mất ngữ cảnh ở rìa; suy luận ảnh lớn ngoài benchmark nên dùng sliding window có overlap hoặc padding.
- IoU/F1 theo pixel chưa phản ánh đầy đủ việc phát hiện từng công trình. Nếu ứng dụng cần đếm công trình, nên bổ sung đánh giá theo đối tượng.
- Tác giả dữ liệu ghi rõ ảnh và nhãn chỉ dành cho **mục đích học thuật**, không dùng thương mại; cần tuân thủ [điều khoản LEVIR-CD](https://github.com/justchenhao/LEVIR#usage).

## 8. Cách trình bày trên CV và khi phỏng vấn

Chỉ dùng bullet có số sau khi chạy thật:

> Built a reproducible satellite building change detection pipeline on LEVIR-CD, comparing an RGB-difference baseline with early-fusion and Siamese U-Nets; achieved **[test F1]** F1 and **[test IoU]** IoU with validation-only threshold selection and pixel-level error analysis.

Nếu chưa chạy đủ, mô tả ở thì hiện tại và không ghi điểm số:

> Developing a LEVIR-CD building change detection benchmark with paired-image U-Nets, split-leakage checks, IoU/F1 evaluation, and visual false-positive/false-negative analysis.

Ba câu nên trả lời được khi phỏng vấn:

1. **Tại sao không dùng accuracy?** Vì lớp không thay đổi áp đảo; dự đoán toàn nền vẫn có accuracy cao nhưng không tìm được công trình thay đổi.
2. **Tại sao ngưỡng chọn ở validation?** Chọn trên test sẽ đưa thông tin từ test vào quyết định mô hình, làm điểm test lạc quan.
3. **Tại sao cần baseline đơn giản?** Để đo xem mạng học được gì vượt lên trên khác biệt màu sắc; nếu mạng không hơn baseline, phải kiểm tra dữ liệu và pipeline trước khi tuyên bố thành công.
