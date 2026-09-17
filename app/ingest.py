import io
import re
from dataclasses import dataclass
from pypdf import PdfReader

@dataclass
class DocumentPage:
    """
    Represents extracted text from a document or specific document page.
    Preserves page numbers for PDF attribution.
    """
    text: str
    page_number: int | None
    source: str

def extract_text_pages(filename: str, file_bytes: bytes) -> list[DocumentPage]:
    """
    Extracts text from PDF or TXT files.
    
    - PDF: Extracts page-by-page, recording 1-indexed page numbers.
    - TXT: Extracts full content with UTF-8 and Latin-1 fallback.
    - Validation: Guards against empty files, corrupt data, and scanned image PDFs.
    
    Raises:
        ValueError: If file type is unsupported, file is empty, or no readable text exists.
    """
    if not file_bytes or len(file_bytes.strip()) == 0:
        raise ValueError(f"Uploaded file '{filename}' is empty (0 bytes).")

    ext = filename.lower().split(".")[-1]
    
    if ext == "txt":
        return _extract_txt(filename, file_bytes)
    elif ext == "pdf":
        return _extract_pdf(filename, file_bytes)
    else:
        raise ValueError(
            f"Unsupported file format: '.{ext}'. "
            "Only PDF (.pdf) and Plain Text (.txt) files are supported."
        )

def _extract_txt(filename: str, file_bytes: bytes) -> list[DocumentPage]:
    """Decodes plain text with fallback for non-UTF8 encodings."""
    try:
        content = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content = file_bytes.decode("latin-1", errors="ignore")
    
    # Normalize excessive whitespace while preserving paragraph breaks
    cleaned = re.sub(r"\r\n|\r", "\n", content).strip()
    
    if len(cleaned) < 20:
        raise ValueError(
            f"Text file '{filename}' contains insufficient readable content "
            "(minimum 20 characters required)."
        )
    
    return [DocumentPage(text=cleaned, page_number=None, source=filename)]

def _extract_pdf(filename: str, file_bytes: bytes) -> list[DocumentPage]:
    """Extracts text page-by-page from PDF files using pypdf."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Failed to parse PDF '{filename}'. File may be corrupted: {str(e)}")
    
    if len(reader.pages) == 0:
        raise ValueError(f"PDF '{filename}' contains 0 pages.")
    
    pages: list[DocumentPage] = []
    total_chars = 0
    
    for idx, page in enumerate(reader.pages):
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""
            
        # Normalize whitespace while preserving line and paragraph structure
        normalized = re.sub(r"[ \t]+", " ", raw_text)
        normalized = re.sub(r"\n\s*\n", "\n\n", normalized).strip()
        
        total_chars += len(normalized)
        if normalized:
            pages.append(
                DocumentPage(
                    text=normalized,
                    page_number=idx + 1,
                    source=filename
                )
            )
            
    if total_chars < 20:
        raise ValueError(
            f"PDF '{filename}' contains no extractable text (extracted {total_chars} characters). "
            "The document is likely scanned or image-only and requires OCR preprocessing."
        )
        
    return pages
