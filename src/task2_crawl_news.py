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


# Nguồn được giao: Học bổng + Sự kiện (RMIT Vietnam)
ARTICLE_URLS = [
    "https://www.rmit.edu.vn/news/all-news/2026/jan/rmit-vietnam-announces-record-2026-scholarships-worth-more-than-200-billion-vnd",
    "https://www.rmit.edu.vn/events/all-events/2026/rmit-tech-camp",
    "https://www.rmit.edu.vn/students/student-news-and-events/student-events-2026/careers-festival",
    "https://www.rmit.edu.vn/sem/discover-rmit-2025-scholarships",
    "https://www.rmit.edu.vn/events/infosessions/ug",
    "https://www.rmit.edu.vn/study-at-rmit/scholarships",
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
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

    # Không lấy raw markdown mặc định: đo trên 6 trang RMIT thì 39-50% mỗi file
    # là thanh menu điều hướng (~9-21K ký tự link trước khi vào nội dung thật).
    # Với chunk_size 500-800 ở Task 4, phần lớn chunk sinh ra sẽ là danh sách
    # link và retrieval trả về toàn menu.
    #
    # Hai cách nhắm theo CSS selector đều KHÔNG dùng được trên trang RMIT:
    #   - css_selector="div.body-gridcontent": class này chỉ có ở template thư
    #     viện; 5/6 trang tin trả về đúng 1 ký tự. Nó còn cắt luôn <head> nên
    #     result.metadata mất sạch, title thành "Unknown".
    #   - target_elements=["div.root"]: div.root của AEM bọc cả header nên nav
    #     vẫn lọt vào markdown.
    # Dùng PruningContentFilter — thuật toán chấm điểm mật độ text/link theo
    # từng khối DOM rồi tỉa khối boilerplate, không phụ thuộc class của template.
    # Đo trên trang tin RMIT: 16,745 -> 5,749 ký tự (giảm 66%), vào thẳng bài viết.
    config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.5, threshold_type="dynamic")
        ),
        excluded_tags=["nav", "footer", "header", "script", "style", "form"],
        exclude_external_links=True,
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config)
        return {
            "url": url,
            "title": result.metadata.get("title", "Unknown"),
            "date_crawled": datetime.now().isoformat(),
            # fit_markdown = bản đã tỉa boilerplate; raw_markdown là bản thô.
            "content_markdown": str(result.markdown.fit_markdown),
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
