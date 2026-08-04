# Bài Tập Nhóm — University Services RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1: Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về dịch vụ và chính sách đại học liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

Xem code mẫu (DeepEval/RAGAS/TruLens) chi tiết trong `README.md` gốc mục "Yêu cầu 2".

### Deliverable Evaluation

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```
┌─ THU THẬP (Task 1-2) ───────────────────────────────────────────────┐
│  rmit.edu.vn                                                         │
│    ├─ Task 1: requests → strip HTML → fpdf2 → 11 PDF chính sách      │
│    │          (trang RMIT là HTML thuần, không publish PDF)          │
│    └─ Task 2: Crawl4AI + PruningContentFilter → 13 bài JSON          │
│               (raw markdown 39-50% là menu → phải tỉa boilerplate)   │
└──────────────────────────┬───────────────────────────────────────────┘
                           ▼  data/landing/{legal,news}/
┌─ CHUẨN HOÁ (Task 3) ────────────────────────────────────────────────┐
│  MarkItDown → 24 file .md trong data/standardized/                   │
│  metadata: source (tên file) + type (legal|news)                     │
└──────────────────────────┬───────────────────────────────────────────┘
                           ▼
┌─ INDEXING (Task 4) ─────────────────────────────────────────────────┐
│  RecursiveCharacterTextSplitter  chunk_size=800  overlap=100         │
│  → 180 chunks → BAAI/bge-m3 (1024d) → ChromaDB (cosine)             │
└──────────────────────────┬───────────────────────────────────────────┘
                           ▼
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
┌─ Task 5: SEMANTIC ──────┐        ┌─ Task 6: LEXICAL ───────┐
│ bge-m3 → ChromaDB       │        │ BM25Okapi               │
│ score = 1 − distance    │        │ k1=1.5  b=0.75          │
│ thang [0,1]             │        │ thang 0 → ~14 không chặn│
└────────────┬────────────┘        └────────────┬────────────┘
             │      cùng một nguồn chunk        │
             └──────────────┬───────────────────┘
                            ▼
┌─ Task 7: RRF FUSION ────────────────────────────────────────────────┐
│  RRF(d) = Σ 1/(60 + rank)                                            │
│  Vì sao RRF: hai thang điểm trên KHÔNG so sánh được với nhau.        │
│  k=60 nén khoảng cách hạng đầu (1/61 vs 1/62) nên một ranker tự tin  │
│  không áp đảo được ranker kia — tài liệu được CẢ HAI đồng thuận thắng.│
└──────────────────────────┬───────────────────────────────────────────┘
                           ▼
┌─ Task 9: PIPELINE + FALLBACK ───────────────────────────────────────┐
│  Ngưỡng so với COSINE GỐC (Task 5), KHÔNG phải điểm RRF —            │
│  điểm RRF top-1 luôn ≈ 1/61 = 0.0164 bất kể có liên quan hay không.  │
│                                                                      │
│  cosine ≥ 0.52  →  trả kết quả hybrid                                │
│  cosine < 0.52  →  Task 8: PageIndex vectorless (11 PDF đã upload)   │
│                    PageIndex rỗng nữa  →  trả [] (không trả rác)      │
│                                                                      │
│  0.52 đo thật: đúng chủ đề ≥ 0.5936 · lạc đề ≤ 0.4518                │
└──────────────────────────┬───────────────────────────────────────────┘
                           ▼
┌─ Task 10: GENERATION ───────────────────────────────────────────────┐
│  _resolve_followup()  ghép ngữ cảnh câu nối tiếp TRƯỚC khi retrieve  │
│  reorder_for_llm()    front + back[::-1] chống lost-in-the-middle    │
│  format_context()     nhúng source để LLM cite được                  │
│  gpt-4o-mini  temp=0.3  top_p=0.9  →  câu trả lời có [Nguồn, Năm]    │
└──────────────────────────┬───────────────────────────────────────────┘
                           ▼
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
┌─ app.py (Streamlit) ────┐        ┌─ evaluation/ (RAGAS) ───┐
│ chat + citation         │        │ 16 câu golden dataset   │
│ slider top_k            │        │ A/B: hybrid vs dense    │
│ expander nguồn + score  │        │ judge: gpt-4o-mini      │
│ conversation memory     │        │ embed: bge-m3 (multiling)│
└─────────────────────────┘        └─────────────────────────┘
```

---

## Phân Công Công Việc

| Thành viên | MSSV | GitHub | Vai trò | Nhiệm vụ | Trạng thái |
|-----------|------|--------|---------|----------|------------|
| Trương Công Thái Đức | 2A202601581 | TruongDuke | Role 1 — Team Leader & RAG Architect | Điều phối; Task 1 (6 PDF thư viện + đăng ký học phần); Task 1-10; kiểm chứng RRF k=60; calibrate ngưỡng fallback 0.52; hợp nhất corpus 3 lần; gom code nhóm vào `main`; conversation memory; báo cáo `results.md` | ✅ 35/35 |
| Trần Trung Hiếu | 2A202602002 | trunghieunef | Role 2 — Data & Pipeline Specialist | Task 1 (3 PDF học phí); Task 4 chunking + ChromaDB; Task 7 RRF; Task 9; Task 10; khử trùng lặp corpus học phí (81% → 0.1%); bổ sung PDF học bổng + ký túc xá | ✅ Task 4-10 xong |
| Phạm Quốc Tuấn | 2A202601983 | phamquoctuan2308 | Role 3 — Frontend & Chatbot Dev | Task 2 (6 bài học bổng + sự kiện); Task 5 semantic search; Task 8 PageIndex; Task 9; Task 10; thiết kế lại giao diện `app.py` | ⚠️ còn `rerank()` ở Task 7 |
| Trần Văn Hiếu | 2A202602030 | Marvis12957 | Role 4 — Evaluation & QA Engineer | Task 2 (5 bài ký túc xá + hỗ trợ SV + thư viện); Task 3 convert Markdown; Task 6 BM25; `golden_dataset.json` 16 câu; `eval_pipeline.py` RAGAS + A/B | ✅ 35/35 |

**Số liệu chốt của nhóm:** 24 documents → 180 chunks · ngưỡng fallback 0.52 · RRF k=60 · 11 PDF đã upload PageIndex

**Phân nhánh:** `main` giữ sản phẩm nhóm (code tốt nhất đã gom, 35/35). Bài cá nhân giữ ở `dev/duc-r1`, `dev/hieu-r2`, `dev/tuan-r3`, `dev/vanhieu-r4` — cả 4 người đều sửa cùng các file `src/task*.py` nên merge hết vào `main` sẽ conflict không gỡ được, và điểm cá nhân phải chấm riêng từng người.

---

## Hướng Dẫn Chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy app
streamlit run app.py
# hoặc
chainlit run app.py
```

---

## Lưu ý

Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
