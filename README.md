# Hệ thống hỗ trợ ra quyết định thu hoạch xoài từ phổ NIR (XAI-driven DSS)

**Tên đề tài do giáo sư giao:** *"Nghiên cứu mô hình hỗ trợ ra quyết định điều hành sản xuất theo thời gian thực tích hợp AI giải thích được (XAI-driven DSS) trong chuỗi chế biến nông sản."*

**Hướng cụ thể (từ 18/10/2026):** *Hệ thống hỗ trợ ra quyết định thu hoạch xoài theo thời gian thực từ phổ NIR, tích hợp AI giải thích được.* Đặc tả đầy đủ: [`docs/PROMPT_ClaudeCode_DSS_XAI_xoai.md`](docs/PROMPT_ClaudeCode_DSS_XAI_xoai.md).

## Bài toán

Hợp tác xã và nhà đóng gói xoài cần quyết định quả nào đủ độ chín để thu hoạch hoặc phân hạng. Độ chín đo bằng hàm lượng chất khô (DMC, %FW). Cắt quả để đo thì chậm và phá mẫu; máy NIR cầm tay cho kết quả tức thì nhưng có sai số, nhất là khi sang mùa vụ mới.

Hệ thống nhận phổ NIR của một quả, dự đoán DMC, rồi đưa ra một trong ba quyết định: **đủ chín**, **chưa chín** hoặc **chưa quyết** (chuyển sang đo phá mẫu). Mỗi quyết định kèm lời giải thích và chỉ số tin cậy.

## Tình trạng hiện tại

- Đã có pilot `scripts/pilot_dss.py` (PLS 11 thành phần, khoảng conformal, hiệu chỉnh lô, cờ bất thường phổ). Các mốc mà pilot đạt được ghi trong mục 5 của đặc tả.
- **Bước 1 (tái hiện baseline PLS) đã xong ngày 18/10/2026**, khớp các mốc. Số liệu trong `results/summary.json`, mục `step1`:

| Tập kiểm thử | Số phổ | RMSEP (%FW) | Bias, dự đoán − thật (%FW) | Mốc |
|---|---|---|---|---|
| Mùa vụ 4 (`Val Ext`) | 1.448 | 1,014 | −0,065 | Anderson 2020: 1,01 và −0,07 |
| Mùa vụ 5 (`Val Ext 2`) | 320 | 1,831 | −1,475 | Pilot: 1,83 và −1,48 |

- **Bước 2 (tầng quyết định với PLS) đã xong ngày 18/10/2026.** Kết quả tái hiện pilot, chỉ lệch 1 quả do khác cách tính ngưỡng conformal (xem `results/step2_decision.json`). Số liệu đầy đủ cho T = 14, 15, 16 %FW trong `results/summary.json`, mục `step2`. Kết quả theo quả, tại T = 15 %FW, khoảng 90% có nửa độ rộng q = 1,255 %FW:

| | Mùa vụ 4 (725 quả) | Mùa vụ 5 (160 quả) |
|---|---|---|
| Không defer: độ đúng | 93,0% | 61,3% |
| Độ phủ thực tế của khoảng 90% | 76,4% | 18,8% |
| Defer theo khoảng conformal: tự quyết / đúng trên quả tự quyết | 71,9% / 99,6% | 46,9% / 72,0% |
| Hiệu chỉnh lô k = 5 + defer: tự quyết / đúng / báo "đủ chín" sai | 71,1% / 99,7% / 0,2% | 53,1% / 95,1% / 4,9% |
| Hiệu chỉnh lô k = 5: độ phủ khoảng 90% | 91,3% | 84,9% |

  Các tỉ lệ có hiệu chỉnh lô là trung bình gộp của 30 lần rút ngẫu nhiên. Độ lệch chuẩn giữa các lần lặp nằm trong `results/step2_decision_metrics.csv`. Ví dụ, mùa vụ 5 với k = 5 có tỉ lệ tự quyết 53,1% ± 6,0%.
- Chưa làm: chọn ngưỡng defer tối ưu theo chi phí (mô-đun M4). Phần này cần các giá trị chi phí có nguồn trích dẫn.
- **Bước 3 (ANN và 1D-CNN) đã xong ngày 24/11/2026.** Ba lần chạy trên Colab (GPU T4): lần 1 (30 mô hình, commit `b59b16d`); lần 2 (15 mô hình AT chạy lại sau khi sửa SNV, 15 seed thêm cho `cnn_TA_cal`, commit `f99661b`); thí nghiệm Keras 2 (10 mô hình `cnn_TA_cal_k2`, commit `d45bc81` cộng một bản vá, xem dưới). Số liệu trong `results/summary.json`, mục `step3`; giá trị từng seed trong `results/step3_regression.csv` và `results/step3_decision_metrics.csv`. Các mốc so sánh lấy từ Phụ lục A của Walsh 2024 (Mendeley Data doi:10.17632/mc9jshwb38.1, `appendix_a_model_results.xlsx`), bias đã đổi sang quy ước dự đoán − thật.

| Cấu hình | Mô tả | Mốc (Walsh 2024, cùng cách chia tập) |
|---|---|---|
| `ann_S_cal` | ANN, đạo hàm bậc 2; huấn luyện trên Cal, dừng sớm theo Tuning | 0,958 (A026; Bảng 5) |
| `ann_S_all` | ANN, đạo hàm bậc 2; huấn luyện trên Cal + Tuning | 0,893 (A002; Bảng 1; Anderson 2020: 0,89) |
| `cnn_TA_cal` | CNN, cắt rồi tăng cường phổ; Cal, dừng sớm theo Tuning | 0,824 (C028; Bảng 5) |
| `cnn_TA_cal_k2` | Như `cnn_TA_cal` nhưng huấn luyện bằng Keras 2 (thí nghiệm) | 0,824 (C028) |
| `cnn_AT_cal` | CNN, tăng cường rồi cắt (cách tiền xử lý tốt nhất của Walsh); Cal, dừng sớm theo Tuning | 0,814 (C030) |
| `cnn_AT_all` | Như trên; huấn luyện trên Cal + Tuning | 0,827 (C006) |
| `cnn_AT_calonly` | Như `cnn_AT_cal` nhưng không dùng Tuning dưới bất kỳ hình thức nào | Dùng cho tầng quyết định: Tuning hiệu chỉnh khoảng conformal |

  RMSEP theo phổ (%FW), trung bình (độ lệch chuẩn) qua các seed. Cột "Seed tốt hơn mốc" đếm số seed có RMSEP mùa vụ 4 thấp hơn mốc của Walsh:

| Cấu hình | Seed | Mùa vụ 4 | Mùa vụ 5 | Bias mùa vụ 5 | Mốc Walsh (mùa vụ 4) | Seed tốt hơn mốc | ±0,05 | Khoảng dự đoán 95% |
|---|---|---|---|---|---|---|---|---|
| PLS (Bước 1) | – | 1,014 | 1,831 | −1,475 | – | – | – | – |
| `ann_S_cal` | 5 | 0,885 (0,029) | 2,575 (0,200) | −2,312 | 0,958 (A026) | 5/5 | không đạt | trong [0,798; 0,972] |
| `ann_S_all` | 5 | 0,918 (0,017) | 2,548 (0,158) | −2,276 | 0,893 (A002) | 0/5 | đạt | trong [0,866; 0,969] |
| `cnn_TA_cal` | 20 | 0,899 (0,046) | 1,303 (0,184) | −0,692 | 0,824 (C028) | 0/20 | không đạt | trong [0,801; 0,997] |
| `cnn_TA_cal_k2` | 10 | 0,883 (0,039) | 1,399 (0,117) | −0,865 | 0,824 (C028) | 0/10 | không đạt | trong [0,789; 0,976] |
| `cnn_AT_cal` | 5 | 0,899 (0,037) | 1,403 (0,296) | −0,844 | 0,814 (C030) | 0/5 | không đạt | trong [0,785; 1,012] |
| `cnn_AT_all` | 5 | 0,939 (0,041) | 1,404 (0,233) | −0,800 | 0,827 (C006) | 0/5 | không đạt | trong [0,815; 1,064] |
| `cnn_AT_calonly` | 5 | 0,913 (0,048) | 1,379 (0,155) | −0,849 | – | – | – | – |

  - Ở mùa vụ 4, cả ANN và CNN đều tốt hơn PLS (0,88–0,94 so với 1,01). Ở mùa vụ 5, CNN tốt hơn hẳn PLS và ANN (1,30–1,40 so với 1,83 của PLS và 2,55–2,58 của ANN), với lệch hệ thống nhỏ hơn nhiều.
  - So với Walsh: cả 6 mốc nằm trong khoảng dự đoán 95%, nhưng chỉ `ann_S_all` đạt tiêu chí ±0,05. Với ANN, mốc của Walsh khi kém hơn mọi seed (A026), khi tốt hơn mọi seed (A002), không theo một chiều. Với CNN, mô hình đơn lẻ của Walsh tốt hơn **mọi** seed của đề tài ở cả bốn cấu hình CNN có mốc (0/20, 0/10, 0/5, 0/5), chênh 0,059–0,112 %FW so với trung bình: đây là khác biệt hệ thống, không phải dao động giữa các seed.

  **Thí nghiệm Keras 2.** Dữ liệu, tiền xử lý, kiến trúc và thiết lập huấn luyện đã đối chiếu là giống Walsh; khác biệt còn lại lớn nhất là Walsh dùng TensorFlow 2.11 (Keras 2), đề tài dùng TensorFlow 2.21 (Keras 3). Cấu hình `cnn_TA_cal_k2` giống hệt `cnn_TA_cal` nhưng huấn luyện bằng Keras 2 (gói `tf-keras` 2.21.0 trên cùng TensorFlow 2.21, chọn qua `TF_USE_LEGACY_KERAS`), 10 seed. RMSEP trung bình (độ lệch chuẩn), Welch t-test (mục `keras_comparison`):

| Tập | Keras 2 (10 seed) | Keras 3 (20 seed) | Chênh lệch | p | Walsh (C028) |
|---|---|---|---|---|---|
| Tuning | 0,809 (0,017) | 0,841 (0,022) | −0,032 | < 0,001 | 0,811 |
| Mùa vụ 4 | 0,883 (0,039) | 0,899 (0,046) | −0,017 | 0,317 | 0,824 |
| Mùa vụ 5 | 1,399 (0,117) | 1,303 (0,184) | +0,097 | 0,092 | – |

  Keras 2 khớp Walsh trên Tuning và khác Keras 3 có ý nghĩa thống kê, nhưng không thu hẹp đáng kể chênh lệch ở mùa vụ 4 (vẫn 0/10 seed tốt hơn C028). **Phiên bản Keras không giải thích được chênh lệch với Walsh ở mùa vụ 4; nguyên nhân chưa xác định.** Các khả năng còn lại không kiểm chứng được: mô hình đơn lẻ của Walsh là một lần chạy thuận lợi; khác biệt khác giữa TensorFlow 2.11 (CPU Intel) và môi trường hiện tại; code lúc Walsh huấn luyện khác code công bố (repo của Walsh chỉ có một commit, tạo sau khi huấn luyện, và notebook công bố không chạy cấu hình C028 vì lệnh `break`). Mô hình chính của đề tài vẫn dùng Keras 3 như kế hoạch ban đầu; thí nghiệm Keras 2 được báo cáo như phân tích độ nhạy. Giới hạn: `tf-keras` 2.21 là nhánh Keras 2 được duy trì cho TensorFlow 2.21, không trùng hoàn toàn với Keras 2.11 (ví dụ EarlyStopping của nó khôi phục trọng số trong `on_train_end` như Keras 3; lớp `EarlyStoppingKeras211` trong `src/models_nn.py` đưa cả hai về cách làm của Keras 2.11). Các mô hình `cnn_TA_cal_k2` được huấn luyện bằng code commit `d45bc81` cộng một bản vá áp ngay trong ô 6b trên Colab (đọc phiên bản Keras từ gói đứng sau `tf.keras`; metadata có trường `code_patch`); commit sau đó chứa bản sửa tương đương (`keras_info`). Metadata ghi gói Keras thực sự được dùng (`keras_package`) và phiên bản của nó.

  Ở lần chạy 2, seed 16–19 của `cnn_TA_cal` chạy trên CPU (một phiên Colab không có GPU). RMSEP mùa vụ 4 của chúng không khác các seed chạy GPU: 0,896 so với 0,900 (n = 4 và 16, Welch p = 0,88; mục `device_comparison`).

  **Độ nhạy theo cách tính SNV của AT** (mục `superseded`): lần chạy 1 tính SNV trên toàn phổ (v1), lần chạy 2 trên 402–1098 nm như file của Walsh (v2). RMSEP trung bình qua 5 seed, v1 → v2:

| Cấu hình | Mùa vụ 4 | Mùa vụ 5 |
|---|---|---|
| `cnn_AT_cal` | 0,869 → 0,899 | 1,104 → 1,403 |
| `cnn_AT_all` | 0,885 → 0,939 | 1,252 → 1,404 |
| `cnn_AT_calonly` | 0,911 → 0,913 | 1,169 → 1,379 |

  Kết quả chính dùng v2 vì đúng phương pháp của Walsh; chọn v1 vì nó tốt hơn trên tập kiểm thử sẽ là chọn mô hình theo tập kiểm thử.

  **Tầng quyết định với CNN** (`cnn_AT_calonly`, T = 15 %FW, trung bình 5 seed; có hiệu chỉnh lô thì gộp 30 lần rút). Mỗi ô ghi PLS (Bước 2) / CNN v2 (độ lệch chuẩn giữa các seed) / CNN v1. Nửa độ rộng khoảng 90%: PLS 1,255; CNN v2 1,280; CNN v1 1,247 %FW.

| Tập | Hiệu chỉnh lô | Độ phủ khoảng 90% | Tự quyết | Đúng trên quả tự quyết | Báo "đủ chín" sai (trên quả tự quyết) |
|---|---|---|---|---|---|
| Mùa vụ 4 | không | 76,4% / 86,2% (2,5) / 84,2% | 71,9% / 72,3% (2,1) / 71,9% | 99,6% / 99,4% (0,2) / 99,2% | 0,2% / 0,1% (0,1) / 0,6% |
| Mùa vụ 4 | k = 5 | 91,3% / 92,2% (1,3) / 92,1% | 71,1% / 71,3% (0,9) / 71,6% | 99,7% / 99,6% (0,2) / 99,7% | 0,2% / 0,3% (0,1) / 0,2% |
| Mùa vụ 5 | không | 18,8% / 68,0% (7,3) / 81,0% | 46,9% / 47,0% (3,2) / 47,4% | 72,0% / 93,5% (3,4) / 97,4% | 0,0% / 0,0% (0,0) / 0,8% |
| Mùa vụ 5 | k = 5 | 84,9% / 83,0% (2,6) / 83,8% | 53,1% / 53,2% (3,3) / 54,6% | 95,1% / 95,5% (1,2) / 96,0% | 4,9% / 3,9% (0,9) / 3,2% |

  - Ở mùa vụ 5 khi không hiệu chỉnh lô, CNN giữ khoảng conformal hữu dụng hơn hẳn PLS (độ phủ 68,0% so với 18,8%; đúng trên quả tự quyết 93,5% so với 72,0%), nhưng độ phủ vẫn xa mức 90%: khoảng conformal không còn hiệu chỉnh đúng khi đổi mùa vụ.
  - Có hiệu chỉnh lô (k = 5), PLS và CNN cho kết quả gần nhau. Với CNN, hiệu chỉnh lô tăng tỉ lệ tự quyết (47,0% → 53,2%) nhưng cũng tăng báo "đủ chín" sai (0,0% → 3,9%). Hai dòng "không" và "k = 5" tính trên tập quả khác nhau (bỏ k quả tham chiếu mỗi quần thể).
  - Số liệu với T = 14 và 16 %FW, và với k = 10, trong `results/summary.json`.

  Đối chiếu với dữ liệu Walsh công bố (Mendeley `mc9jshwb38`), thực hiện ngày 24/11/2026:
  - Dữ liệu, phân chia Cal/Tuning/Val Ext, thứ tự mẫu, nhãn DM và đặc trưng `S` trùng với của đề tài (11.691 phổ mùa vụ 1–4).
  - Mỗi mô hình của Walsh là **một lần huấn luyện duy nhất** (seed 42, TensorFlow 2.11). Trong `raw_results.csv`, A002 và C006 xuất hiện hai lần dưới hai nhãn chia tập nhưng cùng dữ liệu huấn luyện, và kết quả trùng nhau đến 1e-15, tức quá trình huấn luyện của Walsh cho kết quả tất định.
  - Tiền xử lý AT của Walsh (`mishra&passos_2021_pretreatment_adjusted.csv`) tính SNV trên dải **402–1098 nm** với `ddof=1`; code của Walsh không nêu lý do chọn dải này. Lần chạy 1 tính SNV trên toàn phổ 285–1200 nm, nên các mô hình AT được chạy lại (lần 2). Kết quả AT của lần 1 được giữ trong `results/step3/superseded/v1/`.
  - TA của lần chạy 1 tính SNV với `ddof=0`. Ở 3 khối có SNV, giá trị của Walsh bằng giá trị lần chạy 1 nhân √(102/103); 3 khối còn lại trùng. Hằng số nhân này bị bước chuẩn hóa triệt tiêu (sau chuẩn hóa và ép kiểu float32, 13 trên 7,4 triệu giá trị lệch 1 ulp), nên TA không chạy lại. Các seed 5–19 của `cnn_TA_cal` dùng `ddof=1` nhưng vẫn mang nhãn phiên bản đặc trưng 1.
  - Sau khi sửa, `augment` tái tạo các file TA và AT mùa vụ 5 của Walsh đến sai số làm tròn (test `test_augment_reproduces_walsh_input_files`).
  - Từ lần chạy 2, EarlyStopping giả lập Keras 2.11 mà Walsh dùng: chỉ khôi phục trọng số tốt nhất khi dừng sớm; mô hình chạy đủ số epoch tối đa giữ trọng số cuối (Keras 3 khôi phục trong cả hai trường hợp). Ở lần chạy 1 không mô hình nào chạy đủ số epoch tối đa, nên hai cách cho cùng kết quả.

  Hai tiêu chí so sánh: (1) "tương đương" nếu RMSEP trung bình qua các seed lệch mốc không quá 0,05 %FW, khoảng 2 lần độ lệch chuẩn giữa các seed mà Walsh báo cáo (đặt trước khi chạy); (2) giá trị của Walsh nằm trong khoảng dự đoán 95% cho một lần chạy, tính từ các seed của đề tài (thêm sau lần chạy 1, khi xác nhận mốc của Walsh là lần chạy đơn lẻ). Khi viết bài phải báo cáo cả hai tiêu chí và nêu rõ tiêu chí (2) được thêm sau.

  Mốc 0,77 (mùa vụ 4) và 1,18 (mùa vụ 5) trong đặc tả là của mô hình C134. Mô hình này loại thêm 329 phổ ngoại lai theo Mishra & Passos 2021 và chia lại Cal/Tuning ngẫu nhiên theo phổ, nên **không so sánh trực tiếp được** với cách chia tập cố định của đề tài.

  Trong metadata của lần chạy 1 (`results/step3/runs/*.json`), trường `best_epoch` là epoch có loss thấp nhất, **không phải** epoch được EarlyStopping khôi phục. Khi dừng sớm, epoch được khôi phục là epoch cuối cùng làm loss theo dõi giảm hơn `min_delta`, tức 50 epoch trước khi dừng. Hai epoch này khác nhau khi các epoch sau vẫn giảm loss nhưng ít hơn `min_delta`: thường gặp khi theo dõi loss huấn luyện (`min_delta` 10^-2,5), nhưng cũng có khi theo dõi Tuning (ví dụ `ann_S_cal` seed 0: loss thấp nhất ở epoch 475, khôi phục epoch 469). Từ lần chạy 2, metadata ghi `restored_epoch` (epoch của trọng số được giữ), `stopped_early` và `min_monitored_epoch`.

  Phân tích (`src/step3_analysis.py`) chỉ dùng dự đoán có phiên bản đặc trưng hiện hành, ghi `complete`, `missing_runs` và `excluded_older_features` vào `results/step3.json`, và trả mã lỗi nếu còn thiếu mô hình. Trong notebook, script huấn luyện và phân tích được gọi qua `run_module`, nên khi một script dừng vì hết giờ hoặc vì thiếu mô hình, notebook dừng lại và không đóng gói kết quả chưa đầy đủ.

  CLAUDE.md yêu cầu tiền xử lý trên runtime CPU. Notebook chưa làm vậy: đặc trưng được tính ngay trong phiên GPU, vì chỉ mất vài giây.
- **Khảo sát tài liệu về tính mới đã xong (24/11/2026):** xem `docs/literature_novelty_review.md`.
- **Cài lại GC-D (26/11/2026).** Phương pháp group-conformal calibration (GC-D) và biến thể hiệu chỉnh độ lệch GC-D-BC của Bhargav Teja & Divya Meena (2026, *Talanta* 314:130941) được cài lại trong `src/group_conformal.py` theo mục 2.3, mục 4.8 và bảng S1, S22 của bài, vì bài không công bố mã nguồn. Những chỗ bài không nói rõ được đánh dấu `ASSUMPTION` trong code.
  - **Test:** bản cài cho cùng kết quả với một bản cài độc lập (PLS khớp lại cho từng A, T²/Q từ `SpectralOOD`, GLM bằng IRLS tự viết), xem `tests/test_group_conformal.py`.
  - **Kiểm tra trên dữ liệu thật:** tái hiện kịch bản bỏ một mùa vụ của bài trên file v4, đúng dữ liệu bài dùng (85.401 phổ, 10.560 quả, 7 mùa vụ). Chạy bằng `scripts/verify_gcd_talanta.py`, kết quả trong `results/gcd_verification.json` (A ≤ 20, theo bảng S1) và `results/gcd_verification_amax19.json` (A ≤ 19). Độ phủ trung bình theo mùa vụ ở mức danh nghĩa 0,90:

| | Bài báo | A ≤ 20 | A ≤ 19 |
|---|---|---|---|
| GC-D | 0,895 | 0,893 | 0,890 |
| SCP (5 seed) | 0,826 | 0,820 | 0,824 |
| G-W-A | 0,90 | 0,898 | 0,894 |
| ASTM-G | 0,906 | 0,880 (RMSE đều theo phổ) | 0,899 (RMSE cân bằng theo nhóm) |
| Số thành phần PLS | 12–19 | 12–20 | 12–19 |

  Độ phủ khớp trong dung sai ±0,01 đã đặt trước. Có hai điểm phải điều chỉnh so với cách đọc ban đầu, đều do bài mô tả không rõ:
  - Khoảng số thành phần 12–19 của bài chỉ tái hiện được khi tìm A ≤ 19, dù bảng S1 ghi A_max = 20.
  - ASTM-G của bài dùng RMSE cân bằng theo nhóm.

  Mặc định trong `src/group_conformal.py` theo hai điểm này. Interval score vẫn thấp hơn bài: GC-D 5,85 so với 5,90; SCP 6,23 so với 6,43; ASTM-G 5,98 so với 6,18 (với A ≤ 19). Khi chưa có mã gốc, nguyên nhân chưa xác định được; nghi vấn cho SCP là cách lấy tâm khoảng. File v4 chỉ dùng cho phép kiểm tra này, không dùng cho kết quả của đề tài.
- Bước tiếp theo: dùng GC-D và GC-D-BC làm đối chứng trong tầng quyết định (trên file v3); Bước 4 (giải thích quyết định và kiểm chứng độ trung thực của lời giải thích, mô-đun M5).

## Cách chạy

Tạo môi trường ảo một lần (Windows, Python 3.10), từ thư mục gốc repo:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Sau đó chạy bằng Python của venv:

```
.venv\Scripts\python -m pytest                 # kiểm tra dữ liệu, các mốc của Bước 1 và Bước 2 (vài giây)
.venv\Scripts\python -m src.step1_baseline     # Bước 1: in so sánh với các mốc, ghi kết quả vào results/
.venv\Scripts\python -m src.step2_decision     # Bước 2: tầng quyết định, so sánh với pilot, ghi kết quả vào results/
.venv\Scripts\python -m src.step3_analysis     # Bước 3: phân tích dự đoán ANN/CNN đã tải về từ Colab
```

**Bước 3 chạy trên Google Colab** (GPU T4), không chạy trên laptop:

1. Upload `MangoDMC_NIR_Data_v3.csv` vào `MyDrive/Mango_NIR/data/` và mở `notebooks/colab_cnn.ipynb` bằng Colab. Notebook tự lấy mã nguồn từ GitHub, nên mọi thay đổi trong `src/` phải được push trước khi chạy.
2. Chọn *Runtime → Change runtime type → T4 GPU*, rồi chạy lần lượt từng ô. Nếu Colab ngắt kết nối, chạy lại từ đầu: các mô hình đã xong được bỏ qua.
3. Tải `MyDrive/Mango_NIR/step3_results.zip` về và giải nén vào thư mục gốc repo. Thao tác này tạo `results/step3/` và cập nhật `results/summary.json`.

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `data/` | `MangoDMC_NIR_Data_v3.csv` (35 MB, **không commit**, tải theo hướng dẫn ở mục Dữ liệu) |
| `docs/PROMPT_ClaudeCode_DSS_XAI_xoai.md` | Đặc tả đề tài: dữ liệu, kiến trúc M1–M5, các bước thực nghiệm, mốc kiểm tra |
| `docs/literature_novelty_review.md` | Khảo sát tài liệu về tính mới (24/11/2026): các bài đã làm trước, mức độ mới của từng điểm đóng góp, việc cần làm |
| `docs/*.pdf` | Bài báo gốc Anderson 2020 (`1-s2.0-S0925521420301629-main.pdf`), bài Walsh 2024 và luận văn của Walsh (chỉ có trên máy, không commit) |
| `scripts/pilot_dss.py` | Pilot ban đầu, dùng làm tham chiếu |
| `scripts/verify_gcd_talanta.py` | Kiểm tra bản cài lại GC-D bằng cách tái hiện số liệu của Bhargav Teja & Divya Meena (2026) trên file v4 |
| `src/group_conformal.py` | Cài lại GC-D và GC-D-BC (group-conformal calibration theo chẩn đoán PLS), cùng các đối chứng SCP và ASTM-G |
| `src/data.py` | Đọc dữ liệu, tạo mã quả, kiểm tra số phổ và số quả của từng tập |
| `src/preprocess.py` | M1: tiền xử lý phổ (Savitzky–Golay trên toàn phổ, cắt 684–990 nm; tùy chọn SNV; tăng cường phổ 6 dạng cho CNN) |
| `src/models_pls.py` | M2: mô hình PLS và các chỉ số RMSEP, bias, SEP, R² |
| `src/models_nn.py` | M2: ANN và 1D-CNN (Keras, thiết lập theo code của Walsh 2024); chỉ chạy trên Colab |
| `src/uncertainty.py` | M3: khoảng conformal, hiệu chỉnh độ lệch theo lô, điểm bất thường phổ (Q-residual, Hotelling T²) |
| `src/decision.py` | M4: quyết định đủ chín / chưa chín / chưa quyết và các chỉ số ở mức quyết định |
| `src/step1_baseline.py`, `src/step2_decision.py` | Chạy Bước 1, Bước 2 và ghi kết quả |
| `src/step3_train_nn.py`, `src/step3_analysis.py` | Bước 3: huấn luyện ANN/CNN (Colab), phân tích dự đoán (Colab hoặc máy cục bộ) |
| `tests/` | Test cho dữ liệu, tiền xử lý, mô hình, các mô-đun M3 và M4, các mốc của Bước 1, Bước 2, và phần không cần TensorFlow của Bước 3 |
| `results/` | Kết quả: `summary.json` (số liệu tổng hợp), `step1_*`, `step2_*`, `step3*` |
| `notebooks/colab_cnn.ipynb` | Notebook Colab cho Bước 3 |

## Dữ liệu

- **Mango DMC and NIR spectra**, Anderson, Walsh, Subedi, Walsh. Mendeley Data, doi:[10.17632/46htwnp833](https://doi.org/10.17632/46htwnp833) (bản v5), giấy phép CC BY 4.0.
- Bài báo gốc: Anderson et al. (2020), *Postharvest Biology and Technology* 168:111202 (xem mục Tài liệu tham khảo).
- File dùng: `MangoDMC_NIR_Data_v3.csv`, gồm 12.011 phổ và 315 cột (9 cột thông tin, nhãn `DM`, 306 cột phổ hấp thụ 285–1200 nm, bước 3 nm). Tải bộ dữ liệu từ Mendeley Data và đặt file này vào `data/`.
- Chỉ để kiểm tra bản cài lại GC-D: file `MangoDMC_NIR_Data_v4.csv` (260 MB) của cùng bộ dữ liệu Mendeley, đặt vào `data/`. Không dùng cho kết quả của đề tài.
- Không bắt buộc, chỉ để kiểm tra tiền xử lý: file đầu vào mùa vụ 5 đã tiền xử lý của Walsh 2024 (Mendeley Data doi:[10.17632/mc9jshwb38.1](https://doi.org/10.17632/mc9jshwb38.1), thư mục `input/season_5`). Đặt `mishra&passos_2021_pretreatment.csv` và `mishra&passos_2021_pretreatment_adjusted.csv` vào `data/` với tên `Walsh2024_season5_mishra_pretreatment.csv` và `Walsh2024_season5_mishra_pretreatment_adjusted.csv`. Thiếu hai file này thì test tương ứng được bỏ qua. Không dùng chúng để huấn luyện.
- Phân chia có sẵn trong cột `Set`: `Cal`, `Tuning` (huấn luyện); `Val Ext` (mùa vụ 4); `Val Ext 2` (mùa vụ 5). Không chia ngẫu nhiên.

## Tài liệu tham khảo

- Anderson N.T., Walsh K.B., Subedi P.P., Hayes C.H. (2020). Achieving robustness across season, location and cultivar for a NIRS model for intact mango fruit dry matter content. *Postharvest Biology and Technology* 168:111202, doi:10.1016/j.postharvbio.2020.111202. Bài gốc của bộ dữ liệu và baseline PLS.
- Walsh J., Neupane A., Li M. (2024). Evaluation of 1D convolutional neural network in estimation of mango dry matter content. *Spectrochimica Acta Part A* 311:124003, doi:10.1016/j.saa.2024.124003. Code: https://github.com/jemjemwalsh/research-deep-learning-nirs
- Walsh J.P. (2024). *Deep Learning in Estimation of Fruit Attributes Using Near Infrared Spectroscopy.* Luận văn Master of Research, Central Queensland University.

## Yêu cầu và quy ước

- **Giáo sư chỉ chấp nhận bộ dữ liệu công khai đã được các bài báo uy tín công bố hoặc sử dụng.** Không tự thu thập hay tự sinh dữ liệu.
- Huấn luyện mạng nơ-ron (ANN, 1D-CNN) chạy trên **Google Colab (GPU T4)**, không chạy trên laptop. PLS và các kiểm tra nhanh chạy cục bộ được.
- Mọi số liệu trong báo cáo lấy từ `results/summary.json`; chưa chạy thì ghi "chưa có kết quả".
- Văn bản gửi giáo sư viết bằng tiếng Việt, dùng **"Hệ thống hỗ trợ ra quyết định"** (viết đầy đủ, không rút gọn thành "Hệ hỗ trợ").
