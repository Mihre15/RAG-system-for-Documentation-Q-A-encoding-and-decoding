#!/usr/bin/env python3
"""Write docs/backup.pdf.

This file is a sample-data generator. It is not part of the RAG pipeline.
You do not need to explain PDF xref tables in a viva.

What you do need to explain: pypdf reads the text back out, and from
that moment a PDF is just another string, chunked like Markdown.
"""

from __future__ import annotations

from pathlib import Path

SENTENCES = [
    "Automatic cloud backup is turned off by default.",
    "To copy a notebook to a USB drive, run northstar pack backup.nsb.",
    "The pack file is a zip archive of the note text files.",
    "Northstar does not encrypt the pack file.",
    "Restore a pack file with northstar unpack backup.nsb.",
]


def _content_stream(sentences: list[str]) -> bytes:
    """PDF text operators: one Helvetica line per sentence."""
    lines = ["BT", "/F1 12 Tf", "72 700 Td", "18 TL"]
    for index, sentence in enumerate(sentences):
        escaped = (
            sentence.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        )
        if index == 0:
            lines.append(f"({escaped}) Tj")
        else:
            lines.append(f"T* ({escaped}) Tj")
    lines.append("ET")
    return ("\n".join(lines) + "\n").encode("ascii")


def build_pdf(sentences: list[str]) -> bytes:
    """Build a one-page PDF whose text layer is `sentences`."""
    stream = _content_stream(sentences)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        (
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"endstream"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("ascii")
        out += body
        out += b"\nendobj\n"

    xref_at = len(out)
    size = len(objects) + 1
    out += f"xref\n0 {size}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode("ascii")
    return bytes(out)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dest = root / "docs" / "backup.pdf"
    dest.write_bytes(build_pdf(SENTENCES))
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
