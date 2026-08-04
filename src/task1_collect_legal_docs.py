"""
Task 1 — Thu thập văn bản chính sách/quy định dịch vụ đại học.

Mảng phụ trách (Role 1): THƯ VIỆN + ĐĂNG KÝ HỌC PHẦN.

Vì sao phải tự sinh PDF thay vì tải PDF có sẵn:
    Các trang dịch vụ của RMIT Vietnam là HTML thuần, không publish bản PDF.
    Trong khi đó tests/test_individual.py (TestTask1) yêu cầu file .pdf/.docx
    trong data/landing/legal/. Nên pipeline ở đây là:
        HTML  --requests-->  text sạch  --fpdf2-->  PDF
    PDF sinh ra vẫn là "văn bản gốc" theo đúng tinh thần Task 1: nội dung
    nguyên văn từ nguồn công khai, có ghi rõ URL và ngày thu thập ở đầu file.

Chạy:
    python -m src.task1_collect_legal_docs
"""

from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from fpdf import FPDF

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# User-Agent thật: một số trang trường trả 403 cho request không có UA.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    )
}

# Font Unicode — BẮT BUỘC nếu nội dung có tiếng Việt có dấu.
# Font core của fpdf2 (Helvetica) chỉ encode được latin-1 => UnicodeEncodeError.
# Danh sách theo thứ tự ưu tiên, hỗ trợ cả macOS / Linux / Windows.
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",       # Linux
    "C:/Windows/Fonts/arial.ttf",                            # Windows
]

# Nguồn được phân công cho Role 1 (tránh trùng với 3 thành viên còn lại:
# học phí -> Role 2, học bổng -> Role 3, ký túc xá/hỗ trợ SV -> Role 4).
SOURCES = [
    {
        "url": "https://www.rmit.edu.vn/students/my-studies/enrolment",
        "filename": "course-enrolment-rmit.pdf",
        "title": "Course Enrolment - RMIT Vietnam",
    },
    {
        "url": "https://www.rmit.edu.vn/libraryvn/borrowing-and-resources/borrowing-and-returning",
        "filename": "library-borrowing-returning-rmit.pdf",
        "title": "Library Borrowing and Returning - RMIT Vietnam",
    },
    {
        "url": "https://www.rmit.edu.vn/students/support/library-services",
        "filename": "library-services-rmit.pdf",
        "title": "Library Services - RMIT Vietnam",
    },
    {
        "url": "https://www.rmit.edu.vn/libraryvn/about-us/hours-and-locations",
        "filename": "library-hours-locations-rmit.pdf",
        "title": "Library Hours and Locations - RMIT Vietnam",
    },
    {
        "url": "https://www.rmit.edu.vn/libraryvn/borrowing-and-resources/library-resources",
        "filename": "library-resources-rmit.pdf",
        "title": "Library Resources - RMIT Vietnam",
    },
    {
        "url": "https://www.rmit.edu.vn/libraryvn/student-support",
        "filename": "library-student-support-rmit.pdf",
        "title": "Library Student Support - RMIT Vietnam",
    },
]


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def find_unicode_font() -> str | None:
    """Tìm một file TTF Unicode có sẵn trên máy."""
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def fetch_page_text(url: str) -> str:
    """
    Tải trang và bóc lấy phần nội dung chính.

    Bỏ script/style/nav/footer vì chúng chiếm phần lớn HTML của RMIT
    (~278KB/trang) nhưng là menu lặp lại — đưa vào corpus chỉ làm nhiễu
    retrieval ở các task sau.

    Lưu ý: trang RMIT chạy Adobe AEM nên KHÔNG có thẻ <main>, và <article>
    trên một số trang chỉ là một khối phụ vài trăm ký tự. Vì vậy chọn
    container theo LƯỢNG TEXT nhiều nhất thay vì theo thứ tự ưu tiên tag —
    nếu tin thứ tự, trang /students/my-studies/enrolment chỉ bóc ra 242 chars.
    """
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()

    # Thứ tự theo ĐỘ SẠCH giảm dần, lấy cái đầu tiên đủ dài. body là phương án
    # cuối vì nó kéo theo cả thanh nav (RMIT Vietnam / Australia / Europe /
    # Students / Alumni / Staff...) — rác này sẽ chui vào chunk ở Task 4.
    MIN_CONTENT = 800
    best = None
    for cand in [
        soup.find("div", class_="body-gridcontent"),  # container nội dung của AEM
        soup.find("main"),
        soup.find("article"),
        soup.body,
    ]:
        if cand is not None and len(cand.get_text(strip=True)) >= MIN_CONTENT:
            best = cand
            break

    text = best.get_text(separator="\n") if best else soup.get_text(separator="\n")

    # Gộp dòng trống và khoảng trắng thừa do bóc tag để lại.
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def save_as_pdf(title: str, url: str, text: str, output_path: Path, font_path: str | None):
    """Xuất text ra PDF, có header ghi nguồn để trace ngược khi cite."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    if font_path:
        pdf.add_font("Uni", "", font_path)
        pdf.set_font("Uni", size=11)
    else:
        # Không tìm được font Unicode -> ép về latin-1, mất dấu tiếng Việt.
        pdf.set_font("Helvetica", size=11)
        text = text.encode("latin-1", "ignore").decode("latin-1")
        title = title.encode("latin-1", "ignore").decode("latin-1")

    header = (
        f"{title}\n"
        f"Source: {url}\n"
        f"Collected: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"{'-' * 60}\n\n"
    )
    pdf.multi_cell(0, 6, header + text)
    pdf.output(str(output_path))


def collect_all():
    """Tải toàn bộ SOURCES và lưu thành PDF trong data/landing/legal/."""
    setup_directory()

    font_path = find_unicode_font()
    if font_path:
        print(f"✓ Font Unicode: {font_path}")
    else:
        print("⚠ Không tìm thấy font Unicode — tiếng Việt sẽ bị mất dấu trong PDF")

    ok = 0
    for i, src in enumerate(SOURCES, 1):
        print(f"\n[{i}/{len(SOURCES)}] {src['url']}")
        try:
            text = fetch_page_text(src["url"])
            if len(text) < 500:
                print(f"  ⚠ Nội dung quá ngắn ({len(text)} chars) — bỏ qua")
                continue

            output_path = DATA_DIR / src["filename"]
            save_as_pdf(src["title"], src["url"], text, output_path, font_path)

            size_kb = output_path.stat().st_size / 1024
            print(f"  ✓ {src['filename']} ({len(text):,} chars → {size_kb:.1f} KB)")
            ok += 1
        except Exception as e:
            print(f"  ✗ Lỗi: {type(e).__name__}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Hoàn tất: {ok}/{len(SOURCES)} file → {DATA_DIR}")
    print(f"Tiêu chí Task 1: cần ≥3 file  →  {'ĐẠT' if ok >= 3 else 'CHƯA ĐẠT'}")


if __name__ == "__main__":
    collect_all()
