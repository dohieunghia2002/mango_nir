# CLAUDE.md

Hướng dẫn cho Claude Code khi làm việc với repo này. Người dùng có thể sửa file này.

## Tình trạng hiện tại (24/11/2026)

- Mục đích: **viết bài báo** về hệ thống hỗ trợ ra quyết định thu hoạch xoài từ phổ NIR.
- Bước 1 (baseline PLS) và Bước 2 (tầng quyết định với PLS) đã chạy, kết quả trong `results/`. Bước 3 (ANN, 1D-CNN) đã xong ngày 24/11/2026 (lần 1, lần 2 và thí nghiệm Keras 2), kết quả trong `results/summary.json`, mục `step3`. CNN của đề tài kém các mô hình đơn lẻ của Walsh ở mùa vụ 4 một cách hệ thống (0,06–0,11 %FW); phiên bản Keras không giải thích được, nguyên nhân chưa xác định; người dùng đã đồng ý dừng điều tra và báo cáo trung thực. Mô hình chính dùng Keras 3, tiền xử lý AT bản v2. Chi tiết trong `README.md`.
- Khảo sát tài liệu về tính mới đã xong (`docs/literature_novelty_review.md`).
- GC-D và GC-D-BC (Bhargav Teja & Divya Meena 2026) đã được cài lại trong `src/group_conformal.py` và kiểm tra ngày 26/11/2026 bằng cách tái hiện số liệu của bài trên file v4 (`scripts/verify_gcd_talanta.py`, `results/gcd_verification*.json`). Độ phủ khớp ±0,01 khi tìm A ≤ 19 và ASTM-G dùng RMSE cân bằng theo nhóm (mặc định của mô-đun). Interval score thấp hơn bài 1–3%, nguyên nhân chưa rõ.
- Bước 2b (26/11/2026, `src/step2_gcd_baseline.py`, mục `step2_gcd` trong `summary.json`): GC-D/GC-D-BC làm đối chứng trong tầng quyết định trên file v3.
  - **Thiết kế:** hiệu chỉnh theo quả; nhóm theo mùa (chính) và theo quần thể (độ nhạy); cùng các lần rút k quả như Bước 2.
  - **Nhánh `A=11`:** mọi phương pháp dùng chung mô hình Bước 1, để tách ảnh hưởng của mô hình dự đoán.
  - **Kết luận:**
    - Mô hình dự đoán ảnh hưởng lớn khi không hiệu chỉnh lô.
    - Cùng mô hình, GC-D đánh đổi tỉ lệ tự quyết để lấy độ phủ.
    - Có hiệu chỉnh lô k = 5, cách làm của Bước 2 tương đương GC-D-BC nhóm theo mùa.
  - Chi tiết trong README.
- Bước 4a (26/11/2026, `src/explain.py`, `src/step4_explain.py`, mục `step4` trong `summary.json`): giải thích quyết định với PLS.
  - **Kết quả đã đăng ký trước:** độ ổn định tốt; cờ Q/T² không bật cho quả nào. Kiểm tra xóa bằng nội suy (phép kiểm tra chính) không phân biệt được dải quan trọng với dải ngẫu nhiên, vì phép xóa đưa phổ ra ngoài miền dữ liệu (phân tích cơ chế làm sau, mục `post_hoc_interp_mechanism`).
  - **Định vị lại (người dùng đồng ý):** với PLS, lời giải thích là phân tách chính xác; điểm được kiểm chứng là độ ổn định. Kiểm chứng độ trung thực chuyển sang CNN ở Bước 4b (Colab).
- Bước 4b (thiết kế đã duyệt 26/11/2026): code và notebook sẵn sàng (`src/explain_nn.py`, `src/step4b_explain_cnn.py`, `notebooks/colab_explain_cnn.ipynb`), **chưa chạy**. Notebook lấy code từ GitHub, nên người dùng phải push trước khi chạy. Sau khi giải nén `step4b_results.zip`, chạy `.venv/Scripts/python -m src.step4b_explain_cnn --summary-only` để ghi mục `step4b` vào `summary.json`. Không đưa `summary.json` vào file zip, để tránh ghi đè các mục chưa commit. Biện pháp an toàn (27/11/2026): script dừng khi phiên không có GPU (mã 5), khi dự đoán lệch Bước 3 quá 1e-4 %FW (mã 4: người dùng gửi output để quyết định; chạy tiếp bằng `ACCEPT_STEP3_MISMATCH = True`) và khi đã chạy quá 4 giờ (mã 3).
- Việc tiếp theo: người dùng chạy Bước 4b trên Colab; sau đó phân tích kết quả, rồi mô-đun chi phí M4 (cần giá trị chi phí có nguồn trích dẫn). Người dùng tự chạy lệnh git (commit, push); chỉ đưa lệnh git khi người dùng yêu cầu.

## Bối cảnh

- Đề tài do giáo sư giao: *"Nghiên cứu mô hình hỗ trợ ra quyết định điều hành sản xuất theo thời gian thực tích hợp AI giải thích được (XAI-driven DSS) trong chuỗi chế biến nông sản."*
- Hướng cụ thể: **hỗ trợ ra quyết định thu hoạch xoài từ phổ NIR**.
- **Đặc tả đầy đủ của đề tài nằm trong `docs/PROMPT_ClaudeCode_DSS_XAI_xoai.md`. Đọc file này trước khi viết code:** dữ liệu, kiến trúc M1–M5, các bước thực nghiệm, mốc số liệu cần tái hiện, cấu trúc repo mong muốn.
- `README.md` ghi tình trạng hiện tại và cấu trúc thư mục.

## Quy tắc bắt buộc

- **Dữ liệu:** giáo sư chỉ chấp nhận bộ dữ liệu công khai đã được bài báo uy tín công bố hoặc sử dụng. Không tự thu thập, không tự sinh dữ liệu; không dùng dữ liệu chỉ có trên Kaggle, dữ liệu tổng hợp hay preprint không có bài báo gốc. Khi đề xuất dữ liệu, ghi bài báo gốc và tạp chí, và nêu rõ nếu dữ liệu thuộc loại không được chấp nhận. Bộ dữ liệu hiện dùng (Anderson et al. 2020, *Postharvest Biology and Technology*) đạt yêu cầu này.
- **Không bịa số liệu:** mọi con số trong báo cáo phải lấy từ output thực tế (`results/summary.json`). Chưa chạy thì ghi "chưa có kết quả".
- **Tính toán nặng:** không tự huấn luyện mạng nơ-ron (ANN, 1D-CNN) trên máy người dùng; viết notebook Google Colab (GPU T4). Chỉ chạy cục bộ các kiểm tra nhanh (vài giây), và mô tả trước khi chạy. Làm từng bước nhỏ; sau mỗi bước in kết quả kiểm tra (số dòng, số quả, sai số so với mốc) và dừng lại nếu lệch mốc. Không tự làm chuỗi bước lớn ngoài yêu cầu. Lý do: người làm trước từng phản đối khi Claude chạy mô phỏng dài trên laptop mà không hỏi.
- **Giới hạn Colab miễn phí (người dùng ghi nhận ngày 18/10/2026):** mỗi phiên GPU T4 chỉ dùng được tối đa khoảng **5 tiếng**, sau đó có thể bị tạm ngừng cấp GPU. Notebook phải:
  - chạy phần tải và tiền xử lý dữ liệu trên runtime CPU;
  - lưu dữ liệu gọn và checkpoint mỗi epoch lên Drive, chạy tiếp được sau khi bị ngắt;
  - dừng an toàn trước khoảng 4,5 tiếng;
  - đo thời gian một epoch trước khi lên kế hoạch số thí nghiệm.
- **Ngôn ngữ:** mọi tin nhắn hiển thị cho người dùng viết bằng tiếng Việt, kể cả cập nhật ngắn giữa các bước. Code, tên biến, comment viết bằng tiếng Anh. Văn bản gửi giáo sư viết bằng tiếng Việt, dùng **"Hệ thống hỗ trợ ra quyết định"** (viết đầy đủ, không rút gọn thành "Hệ hỗ trợ").
- **Repo công khai:** không commit file PDF, file dữ liệu gốc `data/*.csv` (`.gitignore` đã chặn cả hai), thông tin cá nhân hay khóa API.

## Dữ liệu và mã nguồn

- Dữ liệu: `data/MangoDMC_NIR_Data_v3.csv` (12.011 phổ × 315 cột). Chỉ dùng file v3, không dùng v4 ở giai đoạn này. Giữ nguyên phân chia có sẵn trong cột `Set`, **không chia ngẫu nhiên**.
- **Ngoại lệ (người dùng đồng ý ngày 26/11/2026):** file v4 (`data/MangoDMC_NIR_Data_v4.csv`, cùng bộ Mendeley) chỉ được dùng để **kiểm tra bản cài lại GC-D**, bằng cách tái hiện con số của Bhargav Teja & Divya Meena (2026), vốn dùng file v4. Không dùng v4 cho kết quả của đề tài.
- Mỗi quả được quét 2–2,6 lần và file không có mã quả. Tạo mã quả bằng `groupby(['Set','Pop','Type','Cultivar','DM'])`. RMSEP tính theo phổ (như bài gốc); quyết định đưa ra theo quả (trung bình dự đoán các phổ của quả).
- `scripts/pilot_dss.py` là pilot ban đầu (PLS và tầng quyết định), dùng làm tham chiếu. Mã chính viết trong `src/` theo cấu trúc ở mục 6 của đặc tả. Chạy từ thư mục gốc repo bằng Python của venv: `.venv/Scripts/python -m pytest`, `.venv/Scripts/python -m src.step1_baseline`. **Chỉ cài thư viện vào `.venv/`** (tạo từ `requirements.txt`, phiên bản cố định), không cài vào Python chung của máy; thư viện mới phải thêm vào `requirements.txt` đúng phiên bản đã dùng. Khi console Windows báo lỗi mã hóa tiếng Việt, đặt `PYTHONIOENCODING=utf-8`.
- Ngưỡng conformal q dùng định nghĩa chuẩn: giá trị thứ ⌈(n+1)(1−α)⌉ khi xếp |phần dư| của các quả tập Tuning theo thứ tự tăng dần. Pilot dùng `np.quantile` có nội suy, nên ở mùa vụ 5, T = 15 có đúng 1 quả cho kết quả khác. Test `test_step2_with_pilot_quantile_matches_pilot_exactly` chứng minh đó là khác biệt duy nhất.
- Quy ước bias: `mean(dự đoán − thật)`, giống Anderson 2020 (bias mùa vụ 4 là −0,07). **Walsh 2024 dùng quy ước ngược lại**, `mean(thật − dự đoán)`. Điều này đã xác nhận qua `utils/evaluation.py` trong repo GitHub của Walsh, và qua Bảng 1 của Walsh, nơi PLS của Anderson có bias +0,07. Khi dùng số của Walsh phải đổi dấu bias.
- **Bước 3 (ANN, 1D-CNN) chạy trên Colab** bằng `notebooks/colab_cnn.ipynb`. Notebook clone repo từ GitHub (`REPO_URL` trong notebook, hiện là https://github.com/dohieunghia2002/mango_nir.git), nên phải push code trước khi người dùng chạy. Người dùng upload CSV vào `MyDrive/Mango_NIR/data/`, chạy notebook, rồi giải nén `step3_results.zip` vào thư mục gốc repo. Venv cục bộ không có TensorFlow, nên `src/models_nn.py` chỉ được kiểm tra bằng bước chạy thử (mục 5) trong notebook. Siêu tham số lấy theo code của Walsh (github.com/jemjemwalsh/research-deep-learning-nirs); mốc 0,77 của đặc tả không so sánh được với cách chia tập cố định (xem README). Tiền xử lý TA/AT đã đối chiếu với file đầu vào Walsh công bố (Mendeley `mc9jshwb38`): AT tính SNV trên 402–1098 nm, `ddof=1`. Mỗi mốc của Walsh (Phụ lục A) là một lần chạy đơn lẻ, nên so sánh bằng cả hai tiêu chí ±0,05 so với trung bình và khoảng dự đoán 95% (xem README). Khi đổi đặc trưng của một tiền xử lý, tăng `FEATURE_VERSION` trong `src/step3_train_nn.py` để notebook chạy lại đúng các mô hình bị ảnh hưởng.
- Cố định random seed. Lưu mọi kết quả ra CSV/JSON trong `results/`, kèm `results/summary.json`; dùng file này khi viết báo cáo, không ước lượng số liệu.
- Không tự gán ý nghĩa hóa học cho bước sóng nếu không có nguồn trích dẫn.
- `.gitattributes` đặt `* -text` để giữ nguyên từng byte của file dữ liệu và mô hình.

## Đóng góp của đề tài

Không nằm ở việc dự đoán DMC chính xác hơn (bài toán đã bão hòa, RMSEP 0,77–1,01 %FW), mà ở **tầng quyết định**: biết khi nào "chưa quyết" dựa trên độ bất định đã hiệu chỉnh; xử lý lệch mùa vụ bằng hiệu chỉnh độ lệch theo lô; giải thích từng quyết định và **kiểm chứng độ trung thực của lời giải thích**; đánh giá bằng chỉ số ở mức quyết định. **Khảo sát tài liệu về tính mới (24/11/2026, `docs/literature_novelty_review.md`) đã thay đổi định vị này.** Bhargav Teja & Divya Meena (2026, *Talanta* 314:130941, đã đọc toàn văn) dùng file v4 của cùng bộ dữ liệu xoài và đã làm:
- conformal khi đổi mùa vụ, thiết bị, quần thể (GC-D);
- hiệu chỉnh độ lệch bằng k = 1–10 quả có nhãn, gồm cách dịch tâm cùng ý tưởng với đề tài (nhưng áp trên khoảng GC-D), và biến thể tốt hơn GC-D-BC;
- quyết định "cận dưới ≥ ngưỡng DM" có phân tích tỉ lệ chi phí;
- quy tắc từ chối theo Q;
- thử một CNN.

Kayaoğlu & Erdem (2026, *Applied Sciences*) khôi phục độ phủ bằng 9–15 mẫu có nhãn khi đổi thiết bị. Vì vậy **không được khẳng định "lần đầu"** cho các điểm trên. Đóng góp chính khả dĩ là **giải thích từng quyết định (kể cả "chưa quyết") có kiểm chứng độ trung thực** (Bước 4, chưa có kết quả). Đóng góp phụ: quyết định ba nhánh theo quả có chi phí đo phá mẫu, và xác nhận độc lập các phát hiện của bài trên *Talanta* (họ tự nêu cần xác nhận độc lập). GC-D, GC-D-BC và anchoring phải được dùng làm đối chứng. Tài liệu bổ sung của bài trên *Talanta* (file Word, mục S1–S28) **không có mã nguồn**: phải hỏi tác giả hoặc cài lại theo mục 2.3 và bảng S1 của họ.
