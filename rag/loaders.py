"""Turn Markdown, HTML, and PDF files into plain text.

Analogy: three kinds of boxes arrive. We unpack each box into a pile
of words. Later steps never care what the box looked like.

Markdown is kept as written, headings and all. A heading is a useful
search word ("Install", "Storage"), and we do not need a Markdown parser.

HTML is parsed with the standard-library HTML parser. The head, script,
and style tags are dropped so the page title and page code are not
treated as documentation. A heading is kept on the same card as the
paragraph that follows it.

PDF text is extracted with pypdf. A scanned page with no text layer
becomes an empty document and is skipped, with a warning, at index time.
"""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser
from pathlib import Path

from rag.types import Document

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}

# `head` is skipped so the <title> and embedded page code are not indexed.
# Script and style are listed too, in case they appear in the body.
_SKIP_TAGS = {"script", "style", "head"}
_HEADING_TAGS = {"h1", "h2", "h3"}
_PARAGRAPH_TAGS = {"p", "li"}


class _VisibleText(HTMLParser):
    """Collect the words a reader would see."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        # A heading stays on the same card as the paragraph that follows.
        # A paragraph end is the break between cards.
        if tag in _HEADING_TAGS:
            self._parts.append("\n")
        elif tag in _PARAGRAPH_TAGS:
            self._parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = [line.strip() for line in raw.splitlines()]
        return "\n".join(lines).strip()


def load_html(html: str) -> str:
    """Return the visible text of an HTML document."""
    parser = _VisibleText()
    parser.feed(html)
    parser.close()
    return parser.text()


def load_pdf(path: Path) -> str:
    """Return the text layer of a PDF, one page after another."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency pypdf. From the project folder run: "
            "pip install -r requirements.txt"
        ) from exc
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages).strip()


def load_file(path: Path, source: str | None = None) -> Document:
    """Load one file. `source` is the label we cite later."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = load_pdf(path)
    elif suffix in {".html", ".htm"}:
        text = load_html(path.read_text(encoding="utf-8"))
    elif suffix in {".md", ".markdown", ".txt"}:
        text = path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {path}")
    if source is None:
        source = path.as_posix()
    return Document(source=source, text=text)


def load_directory(docs_dir: Path) -> list[Document]:
    """Load every supported file under docs_dir, in sorted path order.

    Sorted order keeps chunk ids stable from run to run.
    Source labels are relative to the current working directory when
    that is possible, so citations look like docs/storage.md.
    """
    docs_dir = Path(docs_dir)
    if not docs_dir.is_dir():
        raise SystemExit(f"Docs folder not found: {docs_dir}")
    paths = [
        path
        for path in docs_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    cwd = Path.cwd().resolve()
    documents = []
    for path in sorted(paths):
        resolved = path.resolve()
        try:
            source = resolved.relative_to(cwd).as_posix()
        except ValueError:
            source = resolved.as_posix()
        documents.append(load_file(path, source=source))
    return documents


def fingerprint_documents(documents: list[Document]) -> str:
    """Hash the source labels and text.

    Analogy: a wax seal on the box of documents. If the seal changed,
    the card catalog is out of date and must be rebuilt.

    This is a change detector, not a security control.
    """
    digest = hashlib.sha256()
    for doc in sorted(documents, key=lambda item: item.source):
        digest.update(doc.source.encode("utf-8"))
        digest.update(b"\0")
        digest.update(doc.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
