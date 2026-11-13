"""Tests for Markdown, HTML, and PDF loading. No embedding model required."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from rag.chunking import chunk_text
from rag.loaders import fingerprint_documents, load_directory, load_file, load_html
from rag.types import Document
import make_backup_pdf


class LoaderTests(unittest.TestCase):
    def test_html_drops_script_and_style_and_keeps_the_paragraph(self) -> None:
        html = (ROOT / "docs" / "limits.html").read_text(encoding="utf-8")
        text = load_html(html)
        self.assertIn("20000 characters", text)
        self.assertIn("NOTE_TOO_LONG", text)
        self.assertIn("at most 500 notes", text)
        self.assertIn("Export to HTML is not available", text)
        self.assertNotIn("3 megabytes", text)
        self.assertNotIn("stylesheet", text)
        self.assertNotIn("decoy", text)

    def test_html_headings_stay_with_the_following_paragraph(self) -> None:
        html = (ROOT / "docs" / "limits.html").read_text(encoding="utf-8")
        parts = chunk_text(load_html(html))
        note_size = [part for part in parts if "20000 characters" in part]
        self.assertEqual(len(note_size), 1)
        self.assertIn("Note size", note_size[0])
        self.assertTrue(all("3 megabytes" not in part for part in parts))
        self.assertTrue(all(len(part) >= 40 for part in parts))

    def test_directory_skips_unrelated_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.md").write_text(
                "Notebooks live in the Northstar folder for real use.\n",
                encoding="utf-8",
            )
            (root / "skip.py").write_text(
                "maximum photo upload is 3 megabytes\n",
                encoding="utf-8",
            )
            (root / "limits.html").write_text(
                "<html><script>maximum photo upload is 3 megabytes</script>"
                "<p>The maximum note body is 20000 characters.</p></html>",
                encoding="utf-8",
            )
            previous = os.getcwd()
            os.chdir(root)
            try:
                docs = load_directory(root)
            finally:
                os.chdir(previous)
        blob = "\n".join(doc.text for doc in docs)
        sources = {doc.source for doc in docs}
        self.assertNotIn("3 megabytes", blob)
        self.assertIn("20000", blob)
        self.assertIn("Northstar", blob)
        self.assertTrue(all(not source.endswith("skip.py") for source in sources))

    def test_pdf_text_comes_back_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "backup.pdf"
            path.write_bytes(make_backup_pdf.build_pdf(make_backup_pdf.SENTENCES))
            doc = load_file(path, source="docs/backup.pdf")
        for sentence in make_backup_pdf.SENTENCES:
            self.assertIn(sentence, doc.text)

    def test_fingerprint_changes_when_the_text_changes(self) -> None:
        original = [Document("docs/a.md", "hello")]
        edited = [Document("docs/a.md", "hello!")]
        self.assertEqual(
            fingerprint_documents(original),
            fingerprint_documents(list(original)),
        )
        self.assertNotEqual(
            fingerprint_documents(original),
            fingerprint_documents(edited),
        )


if __name__ == "__main__":
    unittest.main()
