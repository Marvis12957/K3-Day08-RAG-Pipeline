# RAG Evaluation Results

## Framework sử dụng

RAGAS 0.1.21 — judge LLM `gpt-4o-mini` (OpenAI), judge embeddings `BAAI/bge-m3` chạy local.

**Corpus:** 24 documents (11 legal PDF + 13 news) → 180 chunks (`chunk_size=800`, `overlap=100`), embedding `BAAI/bge-m3` 1024 chiều, ChromaDB cosine.
**Golden dataset:** 16 câu hỏi tiếng Việt, phủ 4 chủ đề: học phí (4), thư viện (7), đăng ký học phần (3), học bổng + ký túc xá (2).

> ⚠️ **Lưu ý phương pháp — chọn sai embedding cho judge làm hỏng số đo.**
> Lần chạy đầu dùng `all-MiniLM-L6-v2` (chỉ mạnh tiếng Anh) làm judge embeddings và
> cho `answer_relevancy` = **0.309 / 0.281**, thấp bất thường so với dải quen thuộc 0.7–0.9.
> Nguyên nhân: RAGAS tính `answer_relevancy` bằng cách sinh câu hỏi ngược từ answer rồi so
> embedding với câu hỏi gốc — cả hai đều là tiếng Việt, nên model tiếng Anh cho ra điểm gần
> như nhiễu. Đổi sang `bge-m3` (multilingual) thì chỉ số này lên **0.738 / 0.687**.
> Ba chỉ số còn lại gần như không đổi vì chúng dựa nhiều vào LLM judge hơn là embedding.

---

## Overall Scores

| Metric | Config A (hybrid_rerank) | Config B (dense_only) | Δ |
|--------|---------------------------|----------------------|---|
| faithfulness | 0.714 | 0.767 | −0.053 |
| answer_relevancy | 0.738 | 0.687 | **+0.050** |
| context_recall | 0.812 | 0.812 | 0.000 |
| context_precision | 0.704 | 0.828 | **−0.124** |
| **Average** | **0.742** | **0.774** | −0.032 |

---

## A/B Comparison Analysis

**Config A (hybrid_rerank):** semantic search + BM25 → RRF merge (k=60) → rerank → fallback PageIndex khi cosine gốc < 0.52.

**Config B (dense_only):** chỉ semantic search. Giữ nguyên reorder + format_context + LLM generation như Config A để khác biệt duy nhất nằm ở khâu retrieval.

### Kết luận: dense_only thắng về điểm trung bình, nhưng không thắng ở mọi mặt

`dense_only` đạt 0.774 so với 0.742 — cách biệt đến gần như hoàn toàn từ **`context_precision`: 0.828 so với 0.704 (−0.124)**.

Nguyên nhân nằm ở cơ chế BM25. Corpus là tiếng Anh còn câu hỏi là tiếng Việt, nên BM25 chỉ khớp được những token trùng mặt chữ — tên riêng, số, thuật ngữ để nguyên (`RMIT`, `Academic Achievement`, `2026`). Với câu hỏi thuần Việt, nó kéo về chunk chứa từ khoá đúng nhưng ngữ cảnh sai, rồi RRF vẫn cấp cho chúng một suất trong top-k. Đó chính là định nghĩa của precision thấp.

Nhưng đọc riêng `context_precision` là đọc thiếu. `answer_relevancy` của hybrid **cao hơn 0.050** — nghĩa là những chunk BM25 thêm vào, tuy làm loãng context, lại cung cấp mảnh thông tin mà dense bỏ sót, giúp câu trả lời bám sát câu hỏi hơn. `context_recall` hai bên bằng nhau (0.812), cho thấy hybrid không hề lấy thiếu evidence.

**Khuyến nghị cho nhóm:** giữ hybrid nhưng phải thêm bước lọc thật sự sau RRF. Hiện `RERANK_METHOD = "rrf"` nên "rerank" chỉ là gộp thứ hạng, chưa có bước nào chấm lại độ liên quan của từng chunk với câu hỏi. Một cross-encoder đặt sau RRF sẽ cắt đúng phần chunk khớp-từ-khoá-sai-ngữ-cảnh mà vẫn giữ được lợi thế recall của hybrid.

---

## Worst Performers (Bottom 3, Config A)

| # | Question | Faith. | Relev. | Recall | Prec. | Root Cause |
|---|----------|:--:|:--:|:--:|:--:|---|
| 1 | Trường có cung cấp ký túc xá trong khuôn viên không? | 0.000 | 0.000 | 0.000 | 0.000 | Retrieval trượt hoàn toàn — xem phân tích dưới |
| 2 | Học phí hàng năm của chương trình Business là bao nhiêu? | 1.000 | 0.888 | 0.000 | 0.500 | Con số nằm trong bảng, chunking cắt mất quan hệ hàng–cột |
| 3 | Liên hệ thư viện RMIT Hà Nội qua email/điện thoại nào? | 1.000 | 0.822 | 0.000 | 0.700 | Thông tin liên hệ rời rạc, không thành đoạn văn |

### Failure case #1 — mổ xẻ (dùng cho phần demo)

```
Câu hỏi   : "Trường có cung cấp ký túc xá trong khuôn viên không?"
cosine cao nhất : 0.5586      (ngưỡng fallback = 0.52)
5 chunk lấy về  : article_08, article_11, tuition-payment-overview,
                  article_04, fees-and-payments
                  → KHÔNG có article_07, tài liệu duy nhất nói về chỗ ở
```

Nội dung ký túc xá **có trong corpus** (`article_07.md`, 988 ký tự) nhưng không lọt vào top-5. Cả 4 chỉ số đều bằng 0 vì LLM buộc phải trả lời từ context toàn nói về học phí.

Điểm đáng chú ý: cosine 0.5586 chỉ nhỉnh hơn ngưỡng **0.04**. Hệ thống nhận ra đây là câu yếu — nhưng chưa đủ yếu để chuyển sang PageIndex, nên vẫn trả về kết quả hybrid sai. Đây là vùng xám mà một ngưỡng đơn không xử lý được.

**Ba nguyên nhân xếp chồng:**

1. **Tài liệu quá ngắn.** `article_07` chỉ 988 ký tự → đúng 1 chunk. Một chunk đơn độc khó cạnh tranh với tài liệu học phí có hàng chục chunk cùng nói về "sinh viên", "RMIT", "chi phí".
2. **Câu hỏi dạng có/không.** "Trường có cung cấp ký túc xá không?" — câu trả lời đúng là *không, nhưng có hỗ trợ tìm chỗ ở bên ngoài*. Embedding của câu hỏi phủ định không gần với đoạn văn mô tả dịch vụ hỗ trợ.
3. **Ngưỡng đặt hơi thấp cho vùng này.** 0.52 lấy từ điểm giữa khoảng trống đo được (lạc đề cao nhất 0.4518, đúng chủ đề thấp nhất 0.5936). Câu này rơi đúng vào vùng xám giữa hai nhóm.

---

## Recommendations

### Cải tiến 1 — Bổ sung nguồn cho chủ đề ký túc xá
**Action:** Corpus hiện chỉ có `article_07` (988 ký tự) và `accommodation-international-students-rmit.pdf` nói về chỗ ở, trong khi ký túc xá là 1 trong 4 chủ đề chính. Crawl thêm 2–3 trang về housing/accommodation.
**Expected impact:** Failure case #1 được giải quyết tận gốc. Đây là cải tiến rẻ nhất và chắc ăn nhất — không đụng vào code.

### Cải tiến 2 — Cross-encoder rerank sau RRF
**Action:** Đổi `RERANK_METHOD` từ `"rrf"` sang `"cross_encoder"` (Jina reranker v2 multilingual). Hiện sau RRF không có bước nào chấm lại độ liên quan thật của từng chunk.
**Expected impact:** Kéo `context_precision` của hybrid từ 0.704 lên gần mức dense_only (0.828) mà vẫn giữ lợi thế `answer_relevancy` (+0.050). Đây là cách để hybrid thắng dense_only ở cả hai mặt.

### Cải tiến 3 — Chunking theo cấu trúc cho tài liệu dạng bảng
**Action:** Failure #2 và #3 đều là `context_recall = 0` trên nội dung dạng bảng/danh sách liên hệ. `RecursiveCharacterTextSplitter` cắt theo ký tự nên tách rời hàng khỏi tiêu đề cột. Thử `MarkdownHeaderTextSplitter` cho nhóm file legal.
**Expected impact:** `context_recall` tăng ở nhóm câu hỏi tra cứu số liệu — hiện là 2 trong 3 câu tệ nhất.

### Cải tiến 4 — Ngưỡng fallback hai mức
**Action:** Thay một ngưỡng cứng 0.52 bằng hai mức: dưới 0.45 thì fallback thẳng PageIndex; từ 0.45–0.60 thì chạy cả hai rồi lấy kết quả tốt hơn.
**Expected impact:** Bắt được vùng xám như câu 0.5586 ở failure #1, nơi hệ thống "biết mình yếu" nhưng vẫn buộc phải trả kết quả hybrid sai.
