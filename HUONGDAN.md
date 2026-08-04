# Hướng Dẫn Bài Lab Ngày 8 — K3 (University Services RAG)

---

## 1. Bài này là gì?

**K3-Day08-RAG-Pipeline-Starter** là bài lab Ngày 8 — bạn xây dựng một **RAG pipeline end-to-end** (từ thu thập dữ liệu đến chatbot trả lời có trích dẫn nguồn).

Chủ đề của K3: **Dịch vụ & chính sách đại học** — chatbot trả lời câu hỏi về học phí, học bổng, ký túc xá, đăng ký học phần, thư viện, hỗ trợ sinh viên...

Dữ liệu mẫu trong repo được lấy từ trang công khai **RMIT Vietnam** (`rmit.edu.vn`).

Bài này **nối tiếp Ngày 7 (K3 Variant)** — cùng domain "University Services", nên pipeline bạn làm hôm nay sẽ nhất quán với những gì đã học ở lab trước.

---

## 2. Cách làm bài: Nhóm tự chia task + Report cá nhân theo phần mình phụ trách

> **Có, cách này hoàn toàn ổn** — và thực ra **đúng tinh thần bài lab** hơn là bắt 4–6 người làm lại y hệt 10 task.

### Mô hình chung

| Ai | Làm gì | Nộp gì (cá nhân) |
|----|--------|------------------|
| **Cả nhóm** | **Tự họp chia task**, dùng **1 repo chung**, ghép pipeline chạy được | Demo nhóm + chatbot/eval |
| **Mỗi bạn** | **Implement sâu** phần task được giao | **Report cá nhân** + pytest **đúng task của mình** |
| **Không bắt buộc** | Mỗi người tự tay code hết Task 1→10 | — |

Pipeline 10 bước vẫn phải **đủ và chạy được ở cấp nhóm**. Điểm cá nhân căn vào **phần bạn chịu trách nhiệm**, không phải “ai cũng pass 35/35 test”.

### Nguyên tắc coach hay nhắc

1. **Owner rõ ràng** — mỗi file task (`task5_...py`, `task9_...py`...) có **1 người chính**; ghi tên trong `group_project/README.md`.
2. **Hiểu toàn pipeline** — bạn không code Task 7 vẫn phải giải thích được Task 7 làm gì trong demo (ít nhất 2–3 câu).
3. **Report cá nhân ≠ copy code nhóm** — giải thích *lý do kỹ thuật*, *trade-off*, *lỗi đã gặp* ở phần mình làm.
4. **Pytest theo phần mình** — chạy test **đúng task bạn owner**, không cần pass hết 10 task.

### Các bạn tự chia role — không có bảng cố định

**Coach không gán sẵn R1/R2/R3...** Nhóm **tự họp 5–10 phút đầu buổi**, thống nhất ai làm gì dựa trên:
- Sở thích / kinh nghiệm (ai thích crawl data, ai thích UI, ai thích thuật toán...)
- Khối lượng công việc **cân bằng** — tránh 1 người ôm 7 task, 3 người kia nhàn
- Thứ tự phụ thuộc pipeline (data phải xong trước khi index; search xong trước khi ghép Task 9)

**Việc nhóm cần làm ngay:**
1. Mở `group_project/README.md` → điền bảng phân công (tên, MSSV, **task số mấy**, file nào)
2. Chọn 1 người **điều phối** (không nhất thiết = người code nhiều nhất) — nhắc deadline từng giai đoạn
3. Thống nhất **1 repo chung** + cách merge code (ai push branch nào, ai review)

**Gợi ý chia theo “khối”, không bắt buộc:**

| Khối công việc | Task liên quan | Ai thường hợp? |
|----------------|----------------|----------------|
| Data | 1, 2, 3 | Bạn thích crawl, xử lý file |
| Index & search | 4, 5, 6 | Bạn thích ML / embedding |
| Retrieval nâng cao | 7, 8, 9 | Bạn thích ghép pipeline, thuật toán |
| Product | 10, `app.py`, eval | Bạn thích UI, demo, viết báo cáo |

Nhóm 4 người → mỗi người ~2–3 task (+ eval chia chung). Nhóm 6 người → có thể tách eval riêng 1 người. **Cách chia do các bạn quyết**, miễn **đủ 10 task có owner** và **không trùng**.

**Ví dụ thực tế (chỉ tham khảo, không copy):**
- Bạn A: Task 1, 2, 3  
- Bạn B: Task 4, 5  
- Bạn C: Task 6, 7, 8  
- Bạn D: Task 9, 10 + điều phối  
- Bạn E: `app.py`, golden dataset, RAGAS  

Nhóm khác có thể chia hoàn toàn khác — **miễn ghi rõ trong README**.

### Report cá nhân — nộp những gì?

Mỗi bạn nộp **1 file report ngắn** (PDF/MD, 1–2 trang), gồm:

1. **Task mình được nhóm giao** (vd: Task 4, 5 — không cần tên role R1/R2)
2. **Bạn đã implement gì** — file nào, hàm nào, tham số chọn (chunk size, model...)
3. **Vì sao chọn cách đó** — 3–5 câu giải thích kỹ thuật
4. **Kết quả kiểm tra** — screenshot/log pytest task của mình
5. **1 lỗi đã gặp + cách fix** (nếu có)
6. **Cách phần mình nối với task khác trong pipeline** (vd: Task 5 output đi đâu trước Task 9)

**Không cần** viết lại toàn bộ 10 task nếu bạn không làm.

### Chấm điểm cá nhân (theo phần được giao)

| Tiêu chí | Gợi ý trọng số |
|----------|----------------|
| Task mình owner **chạy đúng** (pytest pass) | ~60% |
| **Report cá nhân** rõ ràng, có giải thích kỹ thuật | ~30% |
| Tham gia demo — trả lời được câu hỏi về **phần mình** + hiểu sơ pipeline | ~10% |

Repo starter ghi “35/35 test = 50đ cá nhân” — đó là **mốc nếu làm solo**. Làm theo nhóm: map điểm theo **task bạn owner** (Task 1 = 3đ, Task 4 = 7đ...).

### Lệnh pytest — chạy đúng task của bạn

```bash
# Thay X bằng số task bạn owner, vd Task 5:
pytest tests/test_individual.py::TestTask5 -v

# Nếu owner nhiều task:
pytest tests/test_individual.py::TestTask4 tests/test_individual.py::TestTask5 -v
... (Còn286 dòng dòng)

