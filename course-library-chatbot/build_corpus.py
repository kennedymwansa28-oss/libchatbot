"""
Build the fixed knowledge base for the chatbot.

Reads every PDF in ./knowledge, extracts the text page by page, splits it into
overlapping chunks (remembering the page number), and writes data/corpus.json.gz.

Run this only when you change the documents:   python build_corpus.py
The chatbot itself never modifies the knowledge base.
"""
import gzip
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = ROOT / "knowledge"
OUT_FILE = ROOT / "data" / "corpus.json.gz"

# filename -> (short id, display title)
DOCUMENTS = {
    "1705090770-o-reilly-security-final-ebook.pdf": ("oreilly", "O'Reilly Security Ebook"),
    "Applied_Cryptography_2nd_ed__B__Schneier.pdf": ("applied-crypto", "Applied Cryptography (2nd ed.) - Bruce Schneier"),
    "BEE_049-_design_of_embedded_system.pdf": ("embedded-design", "Design of Embedded Systems (BEE-049)"),
    "Cloud_Computing_Security_Risk_Assessment.pdf": ("cloud-risk", "Cloud Computing: Information Assurance Benefits, Risks and Recommendations (Risk Assessment)"),
    "Cloud_Security_A_Comprehensive_Guide_to_Secure_Cloud_Computing__2010_.pdf": ("cloud-security", "Cloud Security: A Comprehensive Guide to Secure Cloud Computing (2010)"),
    "computer_science_reasoning_about_systems.pdf": ("reasoning", "Computer Science: Reasoning About Systems"),
    "Computer_Security___Principles_-_WILLIAM_STALLINGS_2089.pdf": ("stallings", "Computer Security: Principles and Practice - William Stallings"),
    "Data_Science___Big_Data_Analytics___PDFDrive__.pdf": ("bigdata", "Data Science & Big Data Analytics"),
    "DCAP503_SOFTWARE_TESTING_AND_QUALITY_ASSURANCE.pdf": ("testing", "DCAP503 Software Testing and Quality Assurance"),
    "Digital_Forensics_and_Incident_Response_PDF.pdf": ("forensics", "Digital Forensics and Incident Response"),
    "DownloadFull-TextResearchPaperPDFPp1-14-4.pdf": ("ds-paper", "Data Science and Analytics (Research Book Overview) - M. K. Pasupuleti"),
    "grand_embedded_security_US04.pdf": ("embedded-sec", "Introduction to Embedded Security - Joe Grand (Black Hat USA 2004)"),
    "Mastering_Blockchain_2nd_Edition.pdf": ("blockchain", "Mastering Blockchain (2nd ed.)"),
    "secure_coding_in_c_and_c__.pdf": ("secure-coding", "Secure Coding in C and C++"),
    "The_Mobile_Application_Hacker_s_Handbook_by_Dominic_Chell__Tyrone_Erasmus__Shaun_Colley_and_Ollie_Whitehouse.pdf": ("mobile-hack", "The Mobile Application Hacker's Handbook"),
    "The_Web_Application_Hacker_s_Handbook_-_Finding_and_Exploiting_Security_Flaws__2nd_Edition_by_Dafydd_Stuttard__Marcus_Pinto.pdf": ("web-hack", "The Web Application Hacker's Handbook (2nd ed.)"),
    "The_Cambridge_Handbook_of_the_Law__Ethics_and_Policy_of_Artificial_Intelligence.pdf": ("ai-law", "The Cambridge Handbook of the Law, Ethics and Policy of Artificial Intelligence"),
    "usability.pdf": ("usability", "Usability (Human-Computer Interaction / Usability Engineering)"),
}

CHUNK_CHARS = 1100
OVERLAP_CHARS = 150
MIN_CHUNK_CHARS = 120


def clean_block(text: str) -> str:
    text = text.replace("\u00ad", "")                 # soft hyphen
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)       # de-hyphenate line breaks
    text = re.sub(r"\s*\n\s*", " ", text)              # join lines
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def alpha_ratio(text: str) -> float:
    letters = sum(c.isalpha() for c in text)
    return letters / max(1, len(text))


CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\ufffd]")


def page_paragraphs(page):
    for block in page.get_text("blocks", sort=True):
        if block[6] != 0:      # skip image blocks
            continue
        para = clean_block(block[4])
        if len(para) < 3:
            continue
        # Some PDFs have a broken font encoding that extracts as garbage; drop it.
        if len(CONTROL_CHARS.findall(para)) > 0.01 * len(para):
            continue
        yield CONTROL_CHARS.sub("", para)


def chunk_document(doc_id, pdf_path):
    chunks, current, start_page, end_page = [], "", None, None
    pages = 0

    def flush():
        nonlocal current, start_page, end_page
        text = current.strip()
        if len(text) >= MIN_CHUNK_CHARS and alpha_ratio(text) >= 0.55:
            chunks.append({"doc": doc_id, "page": start_page, "page_end": end_page, "text": text})
        current = current[-OVERLAP_CHARS:] if len(text) > OVERLAP_CHARS else ""
        start_page = end_page

    with fitz.open(pdf_path) as pdf:
        pages = len(pdf)
        for page_no, page in enumerate(pdf, start=1):
            for para in page_paragraphs(page):
                if start_page is None or not current.strip():
                    start_page = page_no
                end_page = page_no
                current = (current + " " + para).strip()
                while len(current) > CHUNK_CHARS * 1.5:   # very long paragraph
                    cut = current.rfind(". ", 0, CHUNK_CHARS)
                    cut = cut + 1 if cut > CHUNK_CHARS // 2 else CHUNK_CHARS
                    piece, current = current[:cut], current[cut - OVERLAP_CHARS:] if cut > OVERLAP_CHARS else current[cut:]
                    if len(piece.strip()) >= MIN_CHUNK_CHARS and alpha_ratio(piece) >= 0.55:
                        chunks.append({"doc": doc_id, "page": start_page, "page_end": end_page, "text": piece.strip()})
                    start_page = end_page
                if len(current) >= CHUNK_CHARS:
                    flush()
        if current.strip():
            flush()
    return chunks, pages


def main():
    missing = [f for f in DOCUMENTS if not (KNOWLEDGE_DIR / f).exists()]
    if missing:
        print("Missing PDFs in ./knowledge:\n  " + "\n  ".join(missing))
        sys.exit(1)

    docs, all_chunks = [], []
    for filename, (doc_id, title) in DOCUMENTS.items():
        chunks, pages = chunk_document(doc_id, KNOWLEDGE_DIR / filename)
        docs.append({"id": doc_id, "title": title, "pages": pages, "chunks": len(chunks)})
        all_chunks.extend(chunks)
        print(f"{doc_id:16s} {pages:5d} pages {len(chunks):6d} chunks")

    OUT_FILE.parent.mkdir(exist_ok=True)
    with gzip.open(OUT_FILE, "wt", encoding="utf-8") as f:
        json.dump({"documents": docs, "chunks": all_chunks}, f, ensure_ascii=False)
    print(f"\nWrote {len(all_chunks):,} chunks to {OUT_FILE} ({OUT_FILE.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
