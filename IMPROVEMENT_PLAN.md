# Kế hoạch cải thiện LEVIR-CD

## Mốc hiện tại

Hai lần chạy early-fusion U-Net dùng cùng cấu hình 30 epoch, seed 42 và split. Lần 1 (khôi phục từ notebook) đạt test F1 0,7369 / IoU 0,5834. Lần 2 (ZIP đầy đủ, có checkpoint) đạt test F1 0,7461 / IoU 0,5951. Chênh lệch **không chứng minh một phương pháp mới tốt hơn**, vì mã và siêu tham số chính không đổi. Lần 2 có `best.pt` để tái dùng và kiểm tra suy luận. Nguồn: [`recovered_results/`](recovered_results/) và [`results/early_run_2/`](results/early_run_2/).

## Quan sát từ lần chạy 2

- Validation F1 trong 10 epoch cuối dao động từ **0,5408 đến 0,7403**; checkpoint tốt nhất ở epoch 29. Mã hiện dùng learning rate cố định 0,001. Dao động này gợi ý thử lịch giảm learning rate, nhưng chưa đủ để kết luận nguyên nhân.
- Test có **FP 1.394.414** và **FN 1.938.993**. Precision 0,7784 cao hơn recall 0,7164; chỉ giảm FP sẽ không giải quyết hết lỗi.
- Trong 2.048 tile test, 1.113 tile không có pixel thay đổi. Chúng tạo 120.448 FP, khoảng **8,6% tổng FP**; phần lớn FP nằm trên tile có thay đổi thật. Vì vậy lấy mẫu thêm tile dương không phải giải pháp hiển nhiên cho FP.
- [`test_82_512_512.png`](results/early_run_2/examples/test_82_512_512.png) cho thấy các công trình lớn bị bỏ sót; [`test_103_768_768.png`](results/early_run_2/examples/test_103_768_768.png) cho thấy thay đổi màu mái nhà bị báo sai. Đây là mô tả hình ảnh, chưa xác nhận nguyên nhân.

## Thứ tự thí nghiệm đề xuất

| Bước | Chỉ thay đổi | Giả thuyết và phép kiểm tra |
|---|---|---|
| 1 | Giảm learning rate từ checkpoint đã lưu, thí dụ 0,0001 trong 10 epoch mới | Xem validation F1 có ổn định và tăng so với 0,7465 tại ngưỡng chọn trên validation hay không. Đây là *fine-tuning* mới, không phải resume chính xác vì checkpoint không chứa trạng thái optimizer. |
| 2 | Thử scheduler giảm learning rate cho một run mới từ đầu | So sánh với một run từ đầu cùng số epoch, kiến trúc, dữ liệu và seed. Không đồng thời đổi augmentation/loss. |
| 3 | Thêm augmentation đổi A/B và biến đổi ánh sáng có kiểm soát **chỉ trên train** | Vì nhãn chỉ biểu thị có/không thay đổi, đổi thứ tự thời điểm giữ nguyên nhãn. Kiểm tra F1, precision/recall và ví dụ thay đổi màu mái. |
| 4 | Thử encoder pretrained trong một kiến trúc mới, ví dụ Siamese ResNet18 | So sánh cả độ chính xác, số tham số và thời gian. Ghi rõ trọng số pretrained và nguồn; không tuyên bố cải thiện của U-Net cũ. |

Chạy thử tốc độ riêng bằng `pin_memory`, `persistent_workers`, số worker và AMP; đo thời gian/epoch và đảm bảo metrics không bị lỗi. Các tối ưu tốc độ **không được tính là cải thiện IoU/F1** nếu chưa đo lại. Những file `dataset (1).py`, `model.py`, `train (1).py` đang ở thư mục gốc là bản thử chưa kiểm chứng; chúng không được đưa vào pipeline chính hay dùng để tuyên bố tốc độ/điểm số. Đặc biệt phải kiểm tra logic split để test không bị dùng làm validation.

## Quy tắc đánh giá

Giữ nguyên train/val/test theo ảnh gốc. Chọn checkpoint, ngưỡng và phương án bằng **validation**; báo lại F1, IoU, precision, recall, TP/FP/FN và thời gian. Vì test LEVIR-CD đã được xem ở hai lần chạy, mọi thử nghiệm tiếp theo trên cùng test nên ghi là *exploratory*. Muốn một ước lượng cuối hoàn toàn độc lập, cần một tập giữ lại mới chưa dùng để ra quyết định, tốt nhất là khu vực/nguồn ảnh khác.

Các công bố có mô hình đạt cao hơn trên LEVIR-CD cho thấy còn dư địa, nhưng cấu hình huấn luyện và giao thức phải được đối chiếu trước khi so sánh trực tiếp. Tham khảo [benchmark IFN của Open-CD](https://github.com/likyoo/open-cd/blob/main/configs/ifn/README.md), [hướng dẫn DataLoader](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html), [AMP của PyTorch](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) và [mô tả dữ liệu của tác giả LEVIR-CD](https://github.com/justchenhao/LEVIR).
