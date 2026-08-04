"""Task 1 - collect RMIT tuition and payment policy pages as PDFs."""

from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from textwrap import wrap

import requests
from fpdf import FPDF

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
FONT_PATH = Path("C:/Windows/Fonts/arial.ttf")

SOURCE_URLS = {
    "tuition-fees-rmit.pdf": [
        "https://www.rmit.edu.vn/study-at-rmit/tuition-fees",
    ],
    "fees-and-payments-rmit.pdf": [
        "https://www.rmit.edu.vn/students/my-studies/fees-and-payments",
    ],
    "tuition-payment-overview-rmit.pdf": [
        "https://www.rmit.edu.vn/study-at-rmit/tuition-fees",
        "https://www.rmit.edu.vn/students/my-studies/fees-and-payments",
    ],
}


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip += 1
        if tag in {"p", "br", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        text = " ".join(data.split())
        if text and not self.skip:
            self.parts.append(text)

    def text(self) -> str:
        lines = [line.strip() for line in "".join(self.parts).splitlines()]
        return "\n".join(line for line in lines if len(line) > 2)


def setup_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_text(url: str) -> str:
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; RAG-lab/1.0)"},
        timeout=30,
    )
    response.raise_for_status()
    parser = TextExtractor()
    parser.feed(response.text)
    return parser.text()


def write_pdf(filename: str, urls: list[str]):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.add_font("ArialUnicode", "", str(FONT_PATH))

    title = filename.removesuffix(".pdf").replace("-", " ").title()
    width = pdf.epw
    pdf.set_font("ArialUnicode", size=16)
    pdf.multi_cell(width, 9, title)
    pdf.set_font("ArialUnicode", size=10)
    pdf.multi_cell(width, 6, f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    pdf.multi_cell(width, 6, "Sources: " + ", ".join(urls))
    pdf.ln(4)

    for url in urls:
        pdf.set_font("ArialUnicode", size=13)
        pdf.multi_cell(width, 8, url)
        pdf.set_font("ArialUnicode", size=10)
        for line in fetch_text(url).splitlines():
            for chunk in wrap(line, width=110) or [""]:
                pdf.multi_cell(width, 5, chunk)
            pdf.ln(1)

    output_path = DATA_DIR / filename
    pdf.output(output_path)
    print(f"Saved: {output_path}")


def download_policy_pdfs():
    setup_directory()
    for filename, urls in SOURCE_URLS.items():
        write_pdf(filename, urls)


if __name__ == "__main__":
    download_policy_pdfs()
