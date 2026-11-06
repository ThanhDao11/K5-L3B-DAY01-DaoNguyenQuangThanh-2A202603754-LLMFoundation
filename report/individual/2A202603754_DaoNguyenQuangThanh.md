# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603754
- **Họ và tên:** Đào Nguyễn Quang Thành
- **SEED** (in ở ô đầu notebook): 8129

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | [x] |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | [x] |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | [x] |

## 2. Kết quả chính

 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            0.500   0.200      0.286        0.519   
2. BoW + MLP (deep learning)         0.590   0.209      0.309        0.400   
3a. Encoder (MiniLM) + kNN           0.884   0.973      0.926        0.947   
3b. Encoder (MiniLM) + LogReg        0.955   0.955      0.955        0.947   
3c. Encoder (MiniLM) + MLP head      0.982   0.973      0.977        0.947   
4a. LLM nhỏ zero-shot                1.000   1.000      1.000        1.000   
4b. LLM lớn zero-shot                0.952   1.000      0.976        0.976   
4c. LLM nhỏ few-shot                 1.000   0.950      0.974        0.974   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)           13.976       4534.368       0.079   
2. BoW + MLP (deep learning)        20.229      15626.042       0.271   
3a. Encoder (MiniLM) + kNN         116.383         15.588      87.535   
3b. Encoder (MiniLM) + LogReg      118.316         40.460      26.015   
3c. Encoder (MiniLM) + MLP head    117.556         37.198      31.123   
4a. LLM nhỏ zero-shot                0.000          0.911     450.604   
4b. LLM lớn zero-shot                0.000          1.177     340.485   
4c. LLM nhỏ few-shot                 0.000          0.964     599.858   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.524           0.0  
2. BoW + MLP (deep learning)          0.699           0.0  
3a. Encoder (MiniLM) + kNN          152.455           0.0  
3b. Encoder (MiniLM) + LogReg        44.656           0.0  
...
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

- Predict("ml_lr"), mình đã dự đoán sai về hình đồ thị của 2 bước, trong thực tế bước của lr=0.001 rất nhanh, đồ thị gần như tuyến tính
- Predict("dl_relu"), F1 sụt giảm rất nhiều khi ReLU được cắt đi, giảm hơn 1/3
- Predict("hard_eval"), mình đã nghĩ prompt injection sẽ làm khó được model AI LLM mình sử dụng, nhưng kết quả trả về gần như tuyệt đối

## 4. Phân tích 50 case khó

Tụt nhiều nhất về số điểm tuyệt đối: BoW + LogReg (-0,45), và nó còn rơi xuống F1 ≈ 0,07, gần như vô dụng. BoW + MLP cũng chỉ còn 0,067. Trong nhóm Embedding, LogReg và MLP head tụt hơn kNN (~0,31 đến 0,34 so với 0,21). LLM tụt ít nhất.

Tập SUB (40 tin) gồm spam và ham điển hình, có từ khoá rõ (“FREE”, “WIN”, “£1000”, “claim now”). Đây đúng là thứ mọi model học tốt. Nó cùng phân phối với dữ liệu train, nên model có thể đã “thuộc” mẫu bề mặt. Điểm cao phản ánh khả năng nhận dạng mẫu quen, chưa phản ánh khả năng hiểu. Nó chỉ có 40 tin, nên ít trường hợp biên: spam giọng tự nhiên, ham chứa từ khoá, né từ khoá, injection. Điểm 1,000 của LLM trên SUB chỉ nói là SUB quá dễ, không nói LLM hoàn hảo.

Model BoW tụt nhiều ở nhóm spam giọng tự nhiên và né từ khóa: Hi, this is Linda from the billing team. Your electricity account shows a small overdue balance, - cơ chế BoR khiến các từ ngữ trong nhóm spam này không bắt được từ, dẫn tới flag nhiều về 0 gây ra false positive.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

Đạt SLA độ trễ (p95<50 ms): 
Các mô hình cục bộ gồm 
1. BoW + LogReg, 
2. BoW + MLP, 
3b. Encoder + LogReg,
3c. Encoder + MLP head đều đạt yêu cầu. 

Đạt cả độ trễ lẫn chất lượng: 
3c. Encoder (MiniLM) + MLP head là tối ưu nhất. Phương án này đáp ứng SLA cực kỳ an toàn ( p95≈43 ms) mà vẫn giữ được chất lượng phân loại vượt trội ( F1≈97.7% , gần tương đương LLM).

## 7. AI log và điều đã học

Mình đã dùng AI để gợi ý về hướng xử lý bài toán thực tế và nhận được cách hướng dẫn để tính toán các mô hình có thỏa mãn không. Lúc đầu mình đã không nghĩ BoW thỏa mãn nhưng AI đã sửa sai và giảng lại cho mình.

Mình nhận ra LLM đã có thể nhận biết được những "chiêu" spam nhanh hơn so với dự kiến của mình. Đồng thời, mình cũng đã biết thêm về quy chuẩn syntax Python, và cách gọi và sự dụng API của LLM.
