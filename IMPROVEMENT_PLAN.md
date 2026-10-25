# Kế hoạch cải thiện LEVIR-CD

## Mốc hiện tại

Hai lần chạy early-fusion U-Net dùng cùng cấu hình 30 epoch, seed 42 và split. Lần 1 (khôi phục từ notebook) đạt test F1 0,7369 / IoU 0,5834. Lần 2 (ZIP đầy đủ, có checkpoint) đạt test F1 0,7461 / IoU 0,5951. Chênh lệch **không chứng minh một phương pháp mới tốt hơn**, vì mã và siêu tham số chính không đổi. Lần 2 có `best.pt` để tái dùng và kiểm tra suy luận. Nguồn: [`recovered_results/`](recovered_results/) và [`results/early_run_2/`](results/early_run_2/).

## Quan sát từ lần chạy 2

- Validation F1 trong 10 epoch cuối dao động từ **0,5408 đến 0,7403**; checkpoint tốt nhất ở epoch 29. Mã hiện dùng learning rate cố định 0,001. Dao động này gợi ý thử lịch giảm learning rate, nhưng chưa đủ để kết luận nguyên nhân.
- Test có **FP 1.394.414** và **FN 1.938.993**. Precision 0,7784 cao hơn recall 0,7164; chỉ giảm FP sẽ không giải quyết hết lỗi.
- Trong 2.048 tile test, 1.113 tile không có pixel thay đổi. Chúng tạo 120.448 FP, khoảng **8,6% tổng FP**; phần lớn FP nằm trên tile có thay đổi thật. Vì vậy lấy mẫu thêm tile dương không phải giải pháp hiển nhiên cho FP.
- [`test_82_512_512.png`](results/early_run_2/examples/test_82_512_512.png) cho thấy các công trình lớn bị bỏ sót; [`test_103_768_768.png`](results/early_run_2/examples/test_103_768_768.png) cho thấy thay đổi màu mái nhà bị báo sai. Đây là mô tả hình ảnh, chưa xác nhận nguyên nhân.

## Nghiên cứu ưu tiên IoU và recall của early-fusion U-Net

Mốc test hiện tại: IoU **0,5951**, F1 **0,7461**, precision **0,7784**, recall **0,7164**, tại ngưỡng 0,30 được chọn trên validation. Audit cho thấy pixel thay đổi chỉ chiếm **4,59% train**, **4,20% validation** và **5,09% test**. Trong 935 tile test có thay đổi, **206 tile có TP = 0**; chúng gây ra 12,3% tổng FN. Các con số này cho thấy mất cân bằng lớp và bỏ sót đối tượng là hai giả thuyết đáng kiểm tra. Đây là **suy luận từ kết quả**, chưa phải chẩn đoán nguyên nhân.

| Ưu tiên | Thí nghiệm trên mô hình cũ | Vì sao thử | Tiêu chí quyết định trên validation |
|---|---|---|---|
| 1 | Quét ngưỡng 0,10–0,50 trên checkpoint hiện tại; lưu bảng precision/recall/IoU/F1 và chọn điểm vận hành trước khi xem test | Không phải train lại; kiểm tra liệu có thể lấy recall cao hơn với mức giảm precision chấp nhận được | Chọn IoU cao nhất hoặc recall >= mục tiêu với IoU tối thiểu đã định. Ngưỡng 0,30 vốn tối ưu F1 validation trong lưới cũ; vì F1 và IoU đơn điệu tương ứng, không kỳ vọng tăng IoU validation chỉ nhờ đổi ngưỡng. |
| 2 | Fine-tune `best.pt` thêm 5–10 epoch với LR 1e-4 rồi 3e-5; hoặc run mới với `ReduceLROnPlateau` | LR cố định 1e-3 và F1 validation dao động lớn ở cuối run | IoU validation và recall cùng tăng, precision không sụt quá mức. Checkpoint cũ thiếu optimizer state: gọi là fine-tune, không phải resume chính xác. |
| 3 | Thử `BCEWithLogitsLoss(pos_weight=2 hoặc 3) + Dice`, giữ mọi thứ khác cố định | Đặt trọng số cao hơn cho pixel dương để phạt FN; tỷ lệ dương train chỉ 4,59% | So với BCE+Dice gốc, đo TP/FP/FN, IoU và recall sau khi mỗi run được chọn ngưỡng trên validation. Không lấy 20,8 (tỷ số âm/dương) làm mặc định vì có thể tăng FP quá mạnh. |
| 4 | Thử `BCE + Tversky` với trọng số FN cao hơn FP, ví dụ mẫu số `TP + 0,3 FP + 0,7 FN` | Cho phép điều chỉnh trực tiếp thiên hướng bỏ sót | Tăng IoU cùng recall trên validation; so với bước 3 để tránh đổi nhiều yếu tố một lúc. |
| 5 | Lấy mẫu train nhiều hơn các tile có thay đổi nhỏ, tối đa khoảng 2 lần; giữ validation/test nguyên | 206 tile dương có TP=0 và median 541 pixel dương, so với median 7.753 ở tile dương có TP>0 | Recall trên nhóm đối tượng nhỏ tăng mà IoU toàn bộ không giảm. Tính nhóm kích thước từ nhãn validation, không chọn theo test. |
| 6 | Bổ sung train-only temporal swap A/B và biến đổi ánh sáng nhẹ, độc lập từng thời điểm | Nhãn LEVIR-CD bao gồm cả công trình xuất hiện và biến mất; A/B swap giữ nhãn nhị phân. Nhiễu ánh sáng giúp kiểm tra FP do khác biệt màu | IoU, recall, precision validation và ảnh lỗi. Chỉ một thay đổi mỗi run; không dùng biến đổi làm lệch vị trí riêng từng ảnh. |

Khi báo cáo, giữ cùng split, epoch budget, batch, seed và cách chọn ngưỡng; chạy ít nhất vài seed cho phương án thắng. IoU và F1 là hai phép biến đổi đơn điệu của cùng TP/FP/FN (`F1 = 2 IoU / (1 + IoU)`), nên không coi chúng là hai tín hiệu độc lập. Để đưa IoU từ 0,595 lên khoảng 0,65 **nếu precision giữ được quanh 0,778**, recall cần lên khoảng **0,80**; đây chỉ là mục tiêu toán học, chưa phải dự báo hiệu quả thí nghiệm. Test hiện đã được xem nhiều lần, vì vậy dùng validation để chọn phương án và ghi rõ test kế tiếp là exploratory.

Nguồn phương pháp: [PyTorch `BCEWithLogitsLoss` và `pos_weight`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html), [PyTorch `ReduceLROnPlateau`](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html), [bài báo Tversky loss](https://arxiv.org/abs/1706.05721), [mô tả LEVIR-CD của tác giả](https://github.com/justchenhao/LEVIR), [đường cong precision–recall trong scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html).

Chạy thử tốc độ riêng bằng `pin_memory`, `persistent_workers`, số worker và AMP; đo thời gian/epoch và đảm bảo metrics không bị lỗi. Các tối ưu tốc độ **không được tính là cải thiện IoU/F1** nếu chưa đo lại. Những file `dataset (1).py`, `model.py`, `train (1).py` ở thư mục gốc thuộc thử nghiệm ResNet34 riêng, chưa được đưa vào pipeline early-fusion U-Net chính. Thử nghiệm này có kết quả validation trong checkpoint nhưng chưa có kết quả test; không trộn các số đó vào bảng trên. Đặc biệt phải kiểm tra logic split để test không bị dùng làm validation.

## Quy tắc đánh giá

Giữ nguyên train/val/test theo ảnh gốc. Chọn checkpoint, ngưỡng và phương án bằng **validation**; báo lại F1, IoU, precision, recall, TP/FP/FN và thời gian. Vì test LEVIR-CD đã được xem ở hai lần chạy, mọi thử nghiệm tiếp theo trên cùng test nên ghi là *exploratory*. Muốn một ước lượng cuối hoàn toàn độc lập, cần một tập giữ lại mới chưa dùng để ra quyết định, tốt nhất là khu vực/nguồn ảnh khác.

Các công bố có mô hình đạt cao hơn trên LEVIR-CD cho thấy còn dư địa, nhưng cấu hình huấn luyện và giao thức phải được đối chiếu trước khi so sánh trực tiếp. Tham khảo [benchmark IFN của Open-CD](https://github.com/likyoo/open-cd/blob/main/configs/ifn/README.md), [hướng dẫn DataLoader](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html), [AMP của PyTorch](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) và [mô tả dữ liệu của tác giả LEVIR-CD](https://github.com/justchenhao/LEVIR).
