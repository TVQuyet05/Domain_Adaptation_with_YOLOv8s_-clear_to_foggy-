# 🧠 Domain Adaptation với YOLOv8s (Clear to Foggy)

## 🎯 1. Đặt Vấn Đề

Trong các bài toán nhận diện vật thể (Object Detection) hiện hành, một thách thức lớn thường gặp là sự sụt giảm hiệu suất đáng kể khi áp dụng mô hình trên môi trường thực tế (Target Domain) không có nhãn hoặc môi trường khác biệt so với tập huấn luyện ban đầu (Source Domain).

Cụ thể, một mô hình được huấn luyện tốt trên điều kiện thời tiết quang đãng (clear) sẽ nhận diện rất kém trong điều kiện có sương mù (foggy) do sự phân phối dữ liệu (data distribution) giữa hai môi trường khác nhau đáng kể. Sương mù làm giảm độ tương phản, mờ nét viền và thay đổi màu sắc của vật thể.

Việc thu thập và gán nhãn lại từ đầu cho điều kiện sương mù là rất tốn kém và mất thời gian. Vì vậy, ta cần tới **Domain Adaptation (DA)** nhằm chuyển giao tri thức từ tập gốc (đã gán nhãn đầy đủ - clear) sang tập đích (không/ít gán nhãn - foggy) để duy trì khả năng nhận diện một cách ổn định nhất mà không cần gán nhãn quy mô lớn cho tập mục tiêu.

---

## ✨ 2. Tinh Chỉnh Mô Hình (YOLOv8 Fine-tuning)

Dự án áp dụng kỹ thuật **Domain-Adversarial Neural Network (DANN)** để tinh chỉnh mô hình **YOLOv8s** cơ sở nhằm thích ứng với điều kiện sương mù:

1. **Cấu trúc DANN với YOLO**: Mở rộng mạng YOLOv8s bằng cách thêm một nhánh phân loại miền (Domain Discriminator). Nhánh này có nhiệm vụ phân biệt xem ảnh đầu vào thuộc về Domain gốc (môi trường clear) hay Domain đích (môi trường foggy).
2. **Gradient Reversal Layer (GRL)**: Trong quá trình huấn luyện bằng lan truyền ngược (backpropagation), thành phần GRL sẽ đảo ngược chiều của gradient từ Domain Discriminator. Việc này buộc bộ trích xuất đặc trưng (Feature Extractor - Backbone) của YOLOv8s phải học ra được những đặc trưng chung, "bất biến" với Domain (domain-invariant features).
3. **Quá trình huấn luyện**:
   - Đầu tiên, YOLO học Object Detection loss thông qua tập ảnh Clear (nguồn có nhãn).
   - Tiếp tục đưa cả ảnh Clear và Foggy qua mô hình vào nhánh Domain Discriminator. Mô hình vừa phải cố gắng nhận diện tốt vật thể ở ảnh gốc, vừa phải "đánh lừa" bộ phận phân biệt xem ảnh đó là có sương mù hay không.

Thông qua việc tinh chỉnh này, YOLOv8 học được các thông tin bất biến (domain-invariant features) giúp model đạt hiệu năng cao trên cả hai domain.

---

## 📊 3. Đánh Giá Kết Quả (mAP50)

Dưới đây là bảng so sánh hiệu suất kiểm thử (độ chính xác trung bình trên ngưỡng IOU 50 - mAP50) để chứng minh tính hiệu quả của mô hình sau khi thực hiện Domain Adaptation. Thông tin đánh giá dựa trên 2 dataset là clear cityscapes và foggy cityscapes:

| Mô hình            | `clear_image` | `foggy_image` |
| ------------------ | :-----------: | :-----------: |
| `model_clear`      |     0.495     |     0.397     |
| `model_foggy`      |     0.425     |     0.482     |
| **ours (DA-YOLO)** |   **0.512**   |   **0.502**   |

**Phân tích kết quả:**

- **`model_clear`**: Mô hình chỉ huấn luyện trên ảnh clear, đạt mAP 0.495 ở ảnh clear, nhưng giảm xuống **0.397** trên ảnh foggy.
- **`model_foggy`**: Chỉ huấn luyện trên ảnh sương mù thì nhận diện khá với môi trường sương (**0.482**), nhưng lại gặp rào cản nhận diện (chỉ mAP **0.425**) khi đánh giá ngược lại trên ảnh clear.
- **Mô hình yolov8_DA (`ours`)**: Bằng cách tiếp cận DANN, mô hình của chúng tôi giải quyết triệt để tính nhạy cảm với sự chuyển biến thời tiết. DA-YOLO cải thiện điểm trên ảnh `clear_image` lên mức **0.512** và **0.502** đối với ảnh `foggy_image`. Cả 2 kết quả này đều vượt qua các mô hình cơ sở độc lập.

_Kết Luận:_ Điều này cho thấy kiến trúc DA-YOLO đã học được các đặc trưng chung bất biến đối với sự khác biệt về hình học không gian (Sương mù), gia tăng hiệu suất cũng như sự phục hồi mô hình trong thời tiết đa dạng một cách nhất quán mà không hề yêu cầu tài nguyên gán nhãn cho tập có sương mù.

---

## 🛠️ Setup Instructions

### 1. Prerequisites

- Python 3.8+
- PyTorch

---

### 2. Install environment.

- Using `conda` to create virtual environment.
