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
- **Bước 3 (ANN và 1D-CNN): đã chạy lần 1 trên Colab ngày 24/11/2026 (30 mô hình, commit `b59b16d`); đang chờ chạy lần 2.** Có 6 cấu hình theo thiết lập của Walsh 2024, mỗi cấu hình 5 seed; `cnn_TA_cal` có thêm 15 seed (5–19). Các mốc so sánh với mùa vụ 4, lấy từ Phụ lục A của Walsh 2024 (Mendeley Data doi:10.17632/mc9jshwb38.1, `appendix_a_model_results.xlsx`), bias đã đổi sang quy ước dự đoán − thật:

| Cấu hình | Mô tả | Mốc (Walsh 2024, cùng cách chia tập) |
|---|---|---|
| `ann_S_cal` | ANN, đạo hàm bậc 2; huấn luyện trên Cal, dừng sớm theo Tuning | 0,958 (A026; Bảng 5) |
| `ann_S_all` | ANN, đạo hàm bậc 2; huấn luyện trên Cal + Tuning | 0,893 (A002; Bảng 1; Anderson 2020: 0,89) |
| `cnn_TA_cal` | CNN, cắt rồi tăng cường phổ; Cal, dừng sớm theo Tuning | 0,824 (C028; Bảng 5) |
| `cnn_AT_cal` | CNN, tăng cường rồi cắt (cách tiền xử lý tốt nhất của Walsh); Cal, dừng sớm theo Tuning | 0,814 (C030) |
| `cnn_AT_all` | Như trên; huấn luyện trên Cal + Tuning | 0,827 (C006) |
| `cnn_AT_calonly` | Như `cnn_AT_cal` nhưng không dùng Tuning dưới bất kỳ hình thức nào | Dùng cho tầng quyết định: Tuning hiệu chỉnh khoảng conformal |

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
- Bước tiếp theo: chạy lại notebook Bước 3 trên Colab (lần 2: 15 mô hình AT và 15 seed thêm cho `cnn_TA_cal`), rồi phân tích kết quả.

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
| `docs/*.pdf` | Bài báo gốc Anderson 2020 (`1-s2.0-S0925521420301629-main.pdf`), bài Walsh 2024 và luận văn của Walsh (chỉ có trên máy, không commit) |
| `scripts/pilot_dss.py` | Pilot ban đầu, dùng làm tham chiếu |
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
