"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trang công khai của một trường đại học.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: thông báo tuyển sinh, sự kiện, dịch vụ thư viện, hỗ trợ sinh viên, học bổng.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Ký túc xá / hỗ trợ sinh viên + thư viện (RMIT Vietnam)
ARTICLE_URLS = [
    "https://www.rmit.edu.vn/student-life/support-services/accommodation",
    "https://www.rmit.edu.vn/students/support/student-connect",
    "https://www.rmit.edu.vn/libraryvn/about-us/news/2025/10-years-book-swap",
    "https://www.rmit.edu.vn/libraryvn/about-us/library-events/2026/rmit-library-seminar-2026",
    "https://www.rmit.edu.vn/students/support/student-academic-success",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, DefaultMarkdownGenerator, PruningContentFilter

    # rmit.edu.vn (AEM) không dùng <nav>/<header>/<footer> chuẩn — mega-menu desktop,
    # thanh search, mobile nav và footer nằm rải rác thành các khối riêng ngay trong
    # <body> (div.header-gridcontent, div.top-nav__search, div[class*=mobinav], div.footer),
    # nên PruningContentFilter một mình không đủ và css_selector cố định 1 khung content
    # không ăn được vì mỗi template site đặt tên khung nội dung khác nhau (đã kiểm tra
    # trang chính vs trang thư viện). Loại trừ đích danh các khối boilerplate này.
    # threshold_type="dynamic" (dựa trên mật độ text/link, tag, độ sâu DOM...) lọc boilerplate
    # hiệu quả hơn nhiều so với "fixed" trên trang này — mega-menu của rmit.edu.vn lặp lại
    # ở nhiều khối rải rác (desktop nav, mobile nav, footer sitemap) mà không dùng thẻ chuẩn
    # <nav>/<footer>, nên chặn từng class cụ thể không xuể. excluded_selector giữ lại như
    # lớp phòng vệ thêm cho các khối chắc chắn không phải nội dung.
    run_config = CrawlerRunConfig(
        excluded_selector="div.header-gridcontent, div.footer, div[aria-label='footer']",
        markdown_generator=DefaultMarkdownGenerator(
            # min_word_threshold=5 cắt luôn cả label ngắn hợp lệ kiểu "**Online Portal**"
            # (2 từ) đứng trước phần mô tả — dùng 2 để giữ label mà vẫn lọc rác 1 từ.
            content_filter=PruningContentFilter(threshold=0.5, threshold_type="dynamic", min_word_threshold=2)
        ),
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=run_config)
        return {
            "url": url,
            "title": result.metadata.get("title", "Unknown"),
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown.fit_markdown,
        }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang thông báo/sự kiện trên trang chính thức của trường đại học")
    else:
        asyncio.run(crawl_all())
