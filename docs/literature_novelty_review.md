# Khảo sát tài liệu về tính mới của đề tài

*Hệ thống hỗ trợ ra quyết định thu hoạch xoài từ phổ NIR. Khảo sát thực hiện ngày 24/11/2026; cập nhật cùng ngày sau khi đọc toàn văn và tài liệu bổ sung của bài trên* Talanta.

## 1. Kết luận chính

1. **Phần lớn các điểm đóng góp dự kiến đã được làm trước, trên cùng họ dữ liệu xoài.** [Bhargav Teja & Divya Meena (2026, *Talanta*)](#tl1) đã đọc toàn văn. Bài dùng file v4 của cùng bộ dữ liệu Mendeley mà đề tài dùng (7 mùa vụ, 31 thiết bị) và đã làm:
   - khoảng dự đoán conformal khi gặp mùa vụ, thiết bị, quần thể mới;
   - hiệu chỉnh độ lệch bằng k = 1–10 quả có nhãn của nhóm mới, gồm cách "dịch tâm khoảng" cùng ý tưởng với cách đề tài đang dùng, và một biến thể tốt hơn (GC-D-BC). Với mùa vụ mới, họ thấy hiệu chỉnh không có lợi khi đã dùng GC-D; nó chỉ có lợi khi độ lệch của nhóm lớn;
   - quyết định "chấp nhận quả nếu cận dưới của khoảng ≥ ngưỡng DM" trên lưới ngưỡng 12–19 %DM, kèm phân tích theo tỉ lệ chi phí giữa chấp nhận nhầm và loại nhầm;
   - một quy tắc từ chối phổ bất thường dựa trên Q;
   - thử một CNN.
2. **[Kayaoğlu & Erdem (2026, *Applied Sciences*)](#tl2)** cho thấy 9–15 mẫu có nhãn ở miền mới đủ khôi phục độ phủ conformal (xét khi đổi thiết bị, không có xoài).
3. **Ý tưởng "từ chối dự đoán khi bất định" cũng đã có**, cho phổ đất ([Schmidinger et al., 2026, preprint](#tl9)) và cho chất khô xoài ([Chen et al., 2022](#tl7)).
4. **Những gì còn lại có khả năng mới**, xếp theo mức độ chắc chắn:
   - **Giải thích từng quyết định, kể cả "chưa quyết", và kiểm chứng độ trung thực của lời giải thích** ở mức quyết định. Bài trên *Talanta* không có XAI. Đây là đóng góp chính khả dĩ nhất, nhưng **chưa có kết quả** (Bước 4).
   - **Quyết định ba nhánh theo quả.** Bài trên *Talanta* chỉ có hai nhánh (chấp nhận / không chấp nhận), tính theo phổ, và không coi đo phá mẫu là một nhánh có chi phí riêng. Đây là mở rộng, không phải ý tưởng mới hoàn toàn.
   - **Xác nhận độc lập.** Bài trên *Talanta* tự nêu ở trang 11 rằng phần hiệu chỉnh độ lệch, phần quyết định và các biến thể thêm sau phản biện "cần xác nhận độc lập", và mô hình học sâu "mới thử một phần (một CNN)". Đề tài có thể đóng góp xác nhận độc lập trên cách chia tập cố định của Anderson 2020 (mùa vụ 4 và 5), với nhiều cấu hình CNN và nhiều seed, ở mức quyết định theo quả.
5. **Hệ quả cho bài báo:** phải chuyển trọng tâm sang tầng giải thích quyết định có kiểm chứng (Bước 4). Phần conformal và hiệu chỉnh độ lệch trở thành nền tảng, trích dẫn và dùng phương pháp của bài trên *Talanta* làm đối chứng (hoặc làm thành phần), không trình bày như đóng góp.

## 2. Phạm vi và cách tìm

- **Câu hỏi:** bốn điểm đóng góp nêu trong `CLAUDE.md` (mục "Đóng góp của đề tài") đã có ai làm chưa:
  - (a) quyết định có lựa chọn "chưa quyết" dựa trên độ bất định đã hiệu chỉnh;
  - (b) xử lý lệch mùa vụ bằng hiệu chỉnh độ lệch theo lô với vài quả đo phá mẫu;
  - (c) giải thích quyết định và kiểm chứng độ trung thực;
  - (d) đánh giá ở mức quyết định.
- **Nguồn:**
  - tìm kiếm web theo từng chủ đề (conformal và NIR, lệch mùa vụ, từ chối dự đoán, XAI cho phổ, hệ thống hỗ trợ ra quyết định thu hoạch xoài);
  - xác minh từng tài liệu (tác giả, tạp chí, năm, DOI, loại công bố) và đọc tóm tắt qua Crossref và OpenAlex;
  - đọc toàn văn và tài liệu bổ sung (file Word, mục S1–S28) của [tl1](#tl1), do người dùng tải về; bài truy cập mở, giấy phép CC-BY.
- **Giới hạn:**
  - Không dùng Scopus/Web of Science.
  - Mã nguồn của [tl1](#tl1) không có trong tài liệu bổ sung, dù mục "Data availability" nói có (xem mục 3).
  - Các tài liệu khác phần lớn chỉ đọc tóm tắt; bảng ở mục 6 ghi rõ mức đã đọc của từng tài liệu.
  - Hai bài quan trọng nhất công bố cách ngày khảo sát 1–2 tuần, nên có thể còn bài mới khác chưa được lập chỉ mục.
  - Preprint được ghi rõ là chưa bình duyệt.

## 3. Bài trên *Talanta*: những gì đã làm (theo toàn văn)

- **Dữ liệu** (trang 3):
  - Mango DMC and NIR spectra, bản 5, **file v4**: 85.401 phổ của 10.560 quả, 7 mùa vụ 2015–2021, 31 thiết bị, 199 quần thể, 10 giống.
  - Đề tài dùng file v3 của cùng bộ dữ liệu, với cách chia tập cố định của Anderson 2020 và mùa vụ 5 của Walsh 2024.
  - Các kịch bản xoài: bỏ một mùa vụ (7 lần), dự báo tiến (mỗi mùa từ các mùa trước), bỏ một thiết bị (24 thiết bị), chia theo quần thể (10 phần).
  - Có kiểm soát rò rỉ theo quả: một quả không bao giờ nằm cả ở tập huấn luyện lẫn tập được dự đoán.
- **Mô hình** (trang 4):
  - PLS1 với tiền xử lý Savitzky–Golay đạo hàm bậc 2, cửa sổ 13, 684–990 nm. Gần giống đề tài, nhưng đề tài dùng cửa sổ 17 theo Anderson 2020.
  - CNN chỉ thử ở mục 4.5 (trang 7), cho kịch bản mùa vụ mới, 3 seed: conformal chia ngẫu nhiên phủ 0,651, hiệu chỉnh ngoài nhóm phủ 0,881, GC-D phủ 0,886.
  - Theo tài liệu bổ sung (bảng S1, S6), CNN này là một mạng PyTorch nhỏ do họ tự thiết kế (1 lớp tích chập 8 bộ lọc, tối đa 40 epoch), không phải CNN của Walsh 2024. RMSE trên mùa vụ mới là 1,403, và σ(x) vẫn lấy từ chẩn đoán PLS.
- **Đánh giá:** độ phủ và interval score **theo phổ**, trung bình qua các nhóm.
- **Kết quả chính** (bảng 2, trang 6; độ phủ ở mức danh nghĩa 0,90):

  | Kịch bản xoài | Conformal chia ngẫu nhiên | GC-D |
  |---|---|---|
  | Bỏ một mùa vụ | 0,826 | 0,895 |
  | Dự báo tiến | 0,783 | 0,877 |
  | Thiết bị mới | 0,794 | 0,879 |

  Phần lớn lợi ích đến từ việc hiệu chỉnh trên phần dư ngoài nhóm; co giãn theo T²/Q giữ độ phủ cho các phổ bất thường.
- **Hiệu chỉnh độ lệch bằng k quả có nhãn** (mục 4.8, trang 8; hình 7c–d, trang 10):
  - Cách "dịch tâm khoảng theo phần dư trung bình của k quả" nâng độ phủ của thiết bị tệ nhất từ 0,356 lên 0,755 với 3 quả, nhưng phủ thừa khi k lớn (0,936 với k = 10).
  - Biến thể **GC-D-BC** mô phỏng bước hiệu chỉnh trong mọi nhóm hiệu chỉnh trước khi ước lượng σ và q, nên giữ độ phủ 0,877–0,886 với mọi k.
  - Với mùa vụ mới, độ lệch nhỏ so với dao động giữa các quả, nên vài quả là chưa đủ (dịch tâm: 0,866 với k = 3).
  - Kết luận của họ: hiệu chỉnh độ lệch chỉ có lợi khi độ lệch của nhóm lớn (thiết bị mới, lô bất thường), và khi đó phải tính đến nhiễu của ước lượng độ lệch.
  - Ví dụ quần thể 114500 (Kensington Pride xanh năm 2020, hái trước khi chín, dự đoán cao 3,33 %DM): GC-D chỉ phủ 0,013; một quả có nhãn với GC-D-BC nâng lên 0,938.
  - Chi tiết trong bảng S22 (20 lần rút, đánh giá trên các quả còn lại). Lưu ý: cả hai cách hiệu chỉnh đều **áp lên khoảng GC-D**, tức khoảng đã nới rộng để chịu lệch mùa vụ.
    - *Mùa vụ mới, bỏ một mùa:* không hiệu chỉnh phủ 0,895. Dịch tâm phủ 0,795 / 0,838 / 0,866 / 0,873 / 0,894 với k = 1 / 2 / 3 / 5 / 10. GC-D-BC phủ 0,872–0,898, với interval score kém hơn không hiệu chỉnh (5,899). Với mùa vụ mới, hiệu chỉnh độ lệch **không có lợi** khi đã dùng GC-D.
    - *Quần thể mới:* không hiệu chỉnh phủ 0,899. Dịch tâm phủ thừa, 0,920–0,966. GC-D-BC giữ 0,900 với mọi k, và khoảng hẹp dần (độ rộng 3,42 → 2,73 %DM khi k = 1 → 10).
- **Quy tắc từ chối và gắn cờ** (bảng S23–S24): gắn cờ khi khoảng rộng ít nhất bằng khoảng 90% giữa của giá trị huấn luyện, hoặc khi Q/Q̃ vượt phân vị 99%. Với kịch bản bỏ một mùa vụ, 3,4% số phổ bị gắn cờ; nhóm bị gắn cờ có độ phủ 0,813 và RMSE 2,68, nhóm còn lại 0,931 và 1,22. Đánh giá bằng độ phủ và RMSE, không bằng độ đúng của quyết định.
- **Quyết định** (mục 4.8, trang 8; hình 8, trang 11):
  - Quy tắc: "chấp nhận quả cho 'DM ≥ L' khi cận dưới của khoảng ≥ L" (khẳng định một phía 95%), trên lưới L = 12–19 %DM, kịch bản mùa vụ mới.
  - Với L = 16 %DM: GC-D chấp nhận 34,2% số quả thực sự đạt, conformal chia ngẫu nhiên 43,4%. Chấp nhận theo giá trị dự đoán giữ được 83,1% số quả đạt, nhưng 14,5% số quả được chấp nhận là không đạt.
  - Phân tích theo tỉ lệ chi phí c giữa chấp nhận nhầm và loại nhầm: GC-D có chi phí kỳ vọng thấp hơn conformal chia ngẫu nhiên khi c > 8,2 ở L = 16 %DM (3,2–37,7 trên cả lưới).
  - Quyết định **hai nhánh, tính theo phổ**.
- **Quy tắc từ chối** (mục 4.5, trang 7): không đưa ra khoảng khi Q/Q̃ vượt phân vị 99% của tập hiệu chỉnh. Quy tắc này từ chối 2,7% phổ mùa vụ mới không bị nhiễu, và kèm cờ "khoảng không hữu ích" ("model not valid for this sample").
- **Không có:** giải thích (XAI); quyết định ba nhánh; đánh giá theo quả; chi phí đo phá mẫu như một nhánh quyết định.
- **Hạn chế tự nêu** (trang 11): mô hình học sâu chỉ thử một phần (một CNN), quyết định chỉ xét cho hai chất phân tích; các biến thể hậu kiểm và phân tích thêm khi phản biện "were designed after seeing the results and need independent confirmation".
- **Mã nguồn:** mục "Data availability" (trang 12) nói mã nguồn, cấu hình và script nằm trong tài liệu bổ sung. Nhưng tài liệu bổ sung tải được chỉ là một file Word (mục S1–S28, 3 hình), không có mã nguồn, file đính kèm hay link. Bảng S1 nhắc tới `config/experiment_config.json` nhưng file này không có. Muốn có mã gốc phải hỏi tác giả; nếu không, cài lại theo mô tả ở mục 2.3 (trang 2) và bảng S1, vốn khá chi tiết.
- **Quyết định tính theo phổ:** bảng S11 có n = 85.401, đúng bằng số phổ.

## 4. Đánh giá từng điểm đóng góp

### 4.1. Khoảng conformal và quyết định "chưa quyết" — mức độ mới: thấp

- Khoảng conformal khi đổi mùa vụ trên dữ liệu xoài, và quyết định theo ngưỡng DM có phân tích chi phí, [tl1](#tl1) đã làm (mục 3).
- Các công trình khác:
  - conformal cho phổ hồng ngoại: [Jović, 2025](#tl11); [Cui et al., 2026, preprint](#tl12);
  - conformal làm hỗ trợ ra quyết định trong nông nghiệp: [Farag et al., 2025](#tl13);
  - độ bất định cho chất khô xoài không dùng conformal: [Wadoux & Ramírez-López, 2025](#tl8); [Chen et al., 2022](#tl7);
  - từ chối rồi đo lại: [Schmidinger et al., 2026, preprint](#tl9).
- **Phần còn khác:** quyết định **ba nhánh** (đủ chín / chưa chín / chưa quyết), trong đó "chưa quyết" là đo phá mẫu hoặc chờ, có chi phí riêng; đánh giá **theo quả**. Đây là mở rộng của quy tắc hai nhánh trong [tl1](#tl1), không đủ làm đóng góp chính.

### 4.2. Hiệu chỉnh độ lệch theo lô với vài quả đo phá mẫu — mức độ mới: rất thấp

- [tl1](#tl1) đã thử k = 1–10 quả có nhãn trên dữ liệu xoài cho mùa vụ, thiết bị và quần thể mới (hình 7d, bảng S22).
  - Cách đề tài đang làm (dịch dự đoán theo độ lệch của k quả, giữ ngưỡng q của tập Tuning) cùng ý tưởng với biến thể "centre shifted" của họ. Khác biệt: họ dịch trên khoảng GC-D, đã nới rộng cho lệch mùa vụ; đề tài dịch trên khoảng hiệu chỉnh từ tập Tuning.
  - Họ đã chỉ ra dịch tâm phủ thừa khi k lớn với quần thể và thiết bị mới, đề xuất GC-D-BC tốt hơn, và cho thấy với mùa vụ mới (độ lệch nhỏ), hiệu chỉnh không có lợi khi đã dùng GC-D.
  - Mùa vụ 5 của đề tài (toàn quả xanh cứng, độ lệch PLS khoảng −1,5 %FW) là trường hợp **độ lệch lớn**. Theo kết luận của chính họ, đây là trường hợp hiệu chỉnh có lợi, tương tự quần thể 114500 của họ.
- [Kayaoğlu & Erdem (2026)](#tl2) làm ý tưởng tương tự (anchoring) cho đổi thiết bị.
- Trong chemometrics, cập nhật mô hình hay điều chỉnh độ lệch khi sang mùa mới vốn quen thuộc:
  - với xoài: [Anderson et al., 2020a](#tl3); [Anderson et al., 2020b](#tl4);
  - với việt quất: [Bai et al., 2022](#tl14).
- Kết quả hiện có của đề tài (mục `step2` và `step3` trong `results/summary.json`), mùa vụ 5, T = 15 %FW:
  - PLS: phủ 18,8% khi không hiệu chỉnh, 84,9% với k = 5.
  - CNN (`cnn_AT_calonly`): phủ 68,0% khi không hiệu chỉnh, 83,0% với k = 5.

  Các kết quả này phù hợp với [tl1](#tl1) nhưng không mới.

### 4.3. Giải thích quyết định và kiểm chứng độ trung thực — mức độ mới: trung bình, đóng góp chính khả dĩ nhất

- **Đã có XAI cho CNN dự đoán chất khô trái cây:** [Passos, 2025](#tl5) (hệ số hồi quy, LIME, SHAP; dải bước sóng quan trọng giống PLS); [Santhosh & Hebri, 2025](#tl16) (SHAP cho xoài).
- **Đã có khuyến nghị kiểm chứng độ trung thực bằng nhiễu loạn hoặc xóa vùng phổ** ([Passos, 2026](#tl6); [Kalatzis & Nega, 2026](#tl17)) và cách gộp bước sóng thành vùng ([Contreras et al., 2024](#tl18)), nhưng các bài này chưa áp dụng cho quyết định.
- **Giải thích vì sao mô hình từ chối** đã có trong học máy nói chung ([Artelt et al., 2022](#tl19); [Artelt et al., 2023](#tl20)), chưa thấy cho phổ NIR.
- **[tl1](#tl1) không có XAI.** Ở cuộc tìm này chưa thấy bài nào giải thích từng quyết định thu hoạch (kể cả "chưa quyết": khoảng cách tới ngưỡng, độ rộng khoảng, cờ phổ bất thường hay lệch lô, vùng bước sóng đóng góp) và kiểm chứng rằng che các vùng được cho là quan trọng làm **quyết định** thay đổi nhiều hơn so với che vùng ngẫu nhiên.
- **Chưa có kết quả.** Kết quả kiểm chứng có thể âm tính và phải được báo cáo như vậy.

### 4.4. Đánh giá ở mức quyết định — mức độ mới: thấp

[tl1](#tl1) đã đánh giá tỉ lệ chấp nhận nhầm và hiệu suất trên lưới ngưỡng; [tl2](#tl2) đánh giá quyết định chấp nhận lô; [Shah et al. (2021)](#tl21) phân loại độ chín xoài trực tiếp. Nên trình bày là cách đánh giá của đề tài.

### 4.5. Phát hiện phụ: tái hiện CNN của Walsh 2024

Kết quả Bước 3: CNN của đề tài kém các mô hình đơn lẻ của [Walsh et al. (2024)](#tl10) ở mùa vụ 4 một cách hệ thống, 0,06–0,11 %FW. Mọi đầu vào đã đối chiếu là giống, và phiên bản Keras không giải thích được chênh lệch. Đây là kết quả phụ về khả năng tái hiện.

## 5. Đề xuất định vị lại và việc cần làm

**Không khẳng định "lần đầu"** cho:
- conformal khi lệch mùa vụ hay đổi thiết bị;
- hiệu chỉnh bằng vài quả có nhãn;
- quyết định theo ngưỡng DM có phân tích chi phí;
- từ chối phổ bất thường.

**Hướng định vị khả dĩ:** "Giải thích quyết định thu hoạch có kiểm chứng độ trung thực, trên nền một tầng quyết định ba nhánh có hiệu chỉnh bất định khi lệch mùa vụ". Các phần đóng góp:
1. **Đóng góp chính (Bước 4):**
   - Lời giải thích cho từng quyết định, gồm "chưa quyết", tách được các lý do: gần ngưỡng, khoảng rộng, phổ bất thường hay lệch lô, vùng bước sóng.
   - Kiểm chứng định lượng độ trung thực ở mức quyết định: che vùng quan trọng so với che vùng ngẫu nhiên; độ ổn định giữa các lần quét của cùng một quả.
   - So sánh PLS và CNN.
2. **Đóng góp phụ:**
   - Mở rộng quy tắc quyết định hai nhánh của [tl1](#tl1) thành ba nhánh theo quả, có chi phí đo phá mẫu (M4).
   - Xác nhận độc lập các phát hiện của [tl1](#tl1) (GC-D, hiệu chỉnh độ lệch bằng k quả) trên cách chia tập cố định của Anderson 2020 và mùa vụ 5, với nhiều cấu hình CNN và nhiều seed. [tl1](#tl1) tự nêu đây là phần cần xác nhận độc lập.

**Việc cần làm:**
1. **Mã nguồn GC-D và GC-D-BC: đã xong (26/11/2026).** Tài liệu bổ sung không có mã, nên đề tài tự cài lại theo mục 2.3 và bảng S1 (`src/group_conformal.py`). Bản cài được kiểm tra bằng cách tái hiện kịch bản bỏ một mùa vụ trên file v4 (`results/gcd_verification*.json`).
   - Độ phủ khớp trong ±0,01: GC-D 0,890 so với 0,895, SCP 0,824 so với 0,826.
   - Phải tìm A ≤ 19 (dù bảng S1 ghi 20) và dùng RMSE cân bằng theo nhóm cho ASTM-G.
   - Interval score thấp hơn bài 1–3%, nguyên nhân chưa rõ.

   Chi tiết trong `README.md`.
2. **Thêm đối chứng vào tầng quyết định:**
   - GC-D thay cho conformal hiệu chỉnh trên Tuning. Cách chia tập cố định chỉ có 3 mùa vụ huấn luyện, nên nhóm có thể là quần thể; điều này cần thiết kế trước khi chạy.
   - GC-D-BC thay cho cách dịch tâm hiện tại.
   - Anchoring của [tl2](#tl2).

   Đây là tính toán nhẹ với PLS. Với CNN, việc tính phần dư ngoài nhóm cần huấn luyện lại theo nhóm trên Colab.
3. **Bước 4 (XAI và kiểm chứng độ trung thực):** ưu tiên cao nhất vì là đóng góp chính.
4. **Mô-đun chi phí M4:** chọn ngưỡng "chưa quyết" theo chi phí, nối tiếp phân tích tỉ lệ chi phí c của [tl1](#tl1). Cần giá trị chi phí có nguồn trích dẫn.
5. **Khi viết bài:** trích dẫn [tl1](#tl1) và [tl2](#tl2) trong phần tổng quan và phần phương pháp, nêu rõ đề tài dùng lại phần nào và khác ở đâu.

## 6. Danh mục tài liệu đã xác minh

Cột "Đã đọc" ghi mức độ đã kiểm tra: **TV** = toàn văn; **TT** = tóm tắt (qua OpenAlex/Crossref); **ĐT** = chỉ đoạn trích của công cụ tìm kiếm.

| Mã | Tài liệu | Loại | Đã đọc |
|---|---|---|---|
| <a id="tl1"></a>tl1 | P.N. Bhargav Teja, S. Divya Meena (2026). Group-conformal calibration of near-infrared prediction intervals for new seasons, instruments and regions using PLS diagnostics. *Talanta* 314:130941. doi:10.1016/j.talanta.2026.130941 | Bài tạp chí, truy cập mở (CC-BY) | TV và tài liệu bổ sung (S1–S28) |
| <a id="tl2"></a>tl2 | M. Kayaoğlu, B. Erdem (2026). Conformal coverage collapse under NIR instrument transfer and the price of restoring it with labelled target samples. *Applied Sciences* 16(22):11417. doi:10.3390/app162211417 | Bài tạp chí, truy cập mở | TT |
| <a id="tl3"></a>tl3 | N.T. Anderson, K.B. Walsh, P.P. Subedi, C.H. Hayes (2020a). Achieving robustness across season, location and cultivar for a NIRS model for intact mango fruit dry matter content. *Postharvest Biology and Technology* 168:111202. doi:10.1016/j.postharvbio.2020.111202 | Bài tạp chí (nguồn dữ liệu của đề tài) | Đã dùng trong đề tài |
| <a id="tl4"></a>tl4 | N.T. Anderson, K.B. Walsh, J.R. Flynn, J. Walsh (2020b). Achieving robustness across season, location and cultivar for a NIRS model for intact mango fruit dry matter content. II. Local PLS and nonlinear models. *Postharvest Biology and Technology*. doi:10.1016/j.postharvbio.2020.111358 | Bài tạp chí | ĐT |
| <a id="tl5"></a>tl5 | D. Passos (2025). Deep tutti-frutti II: Explainability of CNN architectures for fruit dry matter predictions. *Spectrochimica Acta Part A*. doi:10.1016/j.saa.2025.126068 | Bài tạp chí | TT |
| <a id="tl6"></a>tl6 | D. Passos (2026). Convolutional neural networks in Vis–NIR chemometrics: From contradiction to conditional design. *TrAC Trends in Analytical Chemistry*. doi:10.1016/j.trac.2026.119015 | Tổng quan | TT, ĐT (bản arXiv 2605.02636) |
| <a id="tl7"></a>tl7 | X. Chen et al. (2022). Gaussian process regression for prediction and confidence analysis of fruit traits by near-infrared spectroscopy. *Food Quality and Safety*. doi:10.1093/fqsafe/fyac068 | Bài tạp chí | TT |
| <a id="tl8"></a>tl8 | A.M.J.-C. Wadoux, L. Ramírez-López (2025). Uncertainty of predictions in absorption spectroscopy: Modelling with quantile regression forest. *Chemometrics and Intelligent Laboratory Systems*. doi:10.1016/j.chemolab.2025.105473 | Bài tạp chí | TT |
| <a id="tl9"></a>tl9 | J. Schmidinger et al. (2026). Rejections based on predictive uncertainty enable reliable routine soil spectroscopy. arXiv:2606.21179 | Preprint, chưa bình duyệt | TT |
| <a id="tl10"></a>tl10 | J. Walsh, A. Neupane, M. Li (2024). Evaluation of 1D convolutional neural network in estimation of mango dry matter content. *Spectrochimica Acta Part A* 311:124003. doi:10.1016/j.saa.2024.124003 | Bài tạp chí (mốc của đề tài) | Đã dùng trong đề tài |
| <a id="tl11"></a>tl11 | O. Jović (2025). Conformal predictors in chemometric study of mid-infrared food adulteration: quantification of prediction uncertainty. *Food Chemistry*. doi:10.1016/j.foodchem.2025.145387 | Bài tạp chí | TT |
| <a id="tl12"></a>tl12 | H. Cui, Y. Qiao, C. Sun (2026). Aggregated inductive conformal prediction for distribution-free uncertainty quantification in near-infrared spectroscopy: Application to botanical adulteration. SSRN. doi:10.2139/ssrn.7528488 | Preprint, chưa bình duyệt | ĐT |
| <a id="tl13"></a>tl13 | M. Farag, A. Emam, J. Leonhardt, R. Roscher (2025). Enhancing decision support in crop production: Analyzing conformal prediction for uncertainty quantification. *Computers and Electronics in Agriculture*. doi:10.1016/j.compag.2025.110559 | Bài tạp chí | TT |
| <a id="tl14"></a>tl14 | Y. Bai, Y. Fang, B. Zhang, S. Fan (2022). Model robustness in estimation of blueberry SSC using NIRS. *Computers and Electronics in Agriculture*. doi:10.1016/j.compag.2022.107073 | Bài tạp chí | ĐT |
| <a id="tl15"></a>tl15 | M. Yi, Y. Tang, H.J. Wang (2025). Distribution-free prediction sets for regression under target shift. arXiv:2510.10985 | Preprint, chưa bình duyệt | ĐT |
| <a id="tl16"></a>tl16 | S. Santhosh, D. Hebri (2025). Explainable multi-task deep learning and Vis-NIR spectroscopy for non-destructive mango quality assessment. Hội nghị COSMIC (IEEE). doi:10.1109/cosmic67569.2025.11380996 | Bài hội nghị | ĐT |
| <a id="tl17"></a>tl17 | D. Kalatzis, A.D. Nega (2026). Beyond spectral attribution: A validation framework for explainable AI in biomedical spectroscopy. *AI in Medicine*. doi:10.3390/aimed1030025 | Tổng quan | TT |
| <a id="tl18"></a>tl18 | J. Contreras, A. Winterfeld, J. Popp, T.W. Bocklitz (2024). Spectral zones-based SHAP/LIME: Enhancing interpretability in spectral deep learning models through grouped feature analysis. *Analytical Chemistry*. doi:10.1021/acs.analchem.4c02329 | Bài tạp chí | ĐT |
| <a id="tl19"></a>tl19 | A. Artelt, R. Visser, B. Hammer (2022). Model agnostic local explanations of reject. ESANN 2022. doi:10.14428/esann/2022.es2022-34 | Bài hội nghị | ĐT |
| <a id="tl20"></a>tl20 | A. Artelt, R. Visser, B. Hammer (2023). "I do not know! but why?" — Local model-agnostic example-based explanations of reject. *Neurocomputing*. doi:10.1016/j.neucom.2023.126722 | Bài tạp chí | Chỉ thông tin thư mục |
| <a id="tl21"></a>tl21 | S.S.A. Shah et al. (2021). Mango maturity classification instead of maturity index estimation: A new approach towards handheld NIR spectroscopy. *Infrared Physics & Technology* 115:103639. doi:10.1016/j.infrared.2021.103639 | Bài tạp chí | ĐT |

Tài liệu tham khảo thêm về bối cảnh thu hoạch xoài (đọc tóm tắt): H.K. Dhonju et al. (2024), Management Information Systems for Tree Fruit–2: Design of a Mango Harvest Forecast Engine, *Horticulturae* 10(3):301, doi:10.3390/horticulturae10030301; S. Tonetto de Freitas et al. (2022), Mango dry matter content at harvest to achieve high consumer quality of different cultivars in different growing seasons, *Postharvest Biology and Technology*, doi:10.1016/j.postharvbio.2022.111917.
