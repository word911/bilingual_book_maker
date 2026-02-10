import pickle
from pathlib import Path

import pytest

ebooklib = pytest.importorskip("ebooklib")
from ebooklib import epub

from book_maker.loader.epub_loader import EPUBBookLoader


class ResumeFlakyModel:
    fail_enabled = True
    fail_on_call = 3
    call_count = 0

    def __init__(
        self,
        key,
        language,
        api_base=None,
        context_flag=False,
        context_paragraph_limit=0,
        temperature=1.0,
        source_lang="auto",
        **kwargs,
    ):
        pass

    @classmethod
    def reset(cls, fail_enabled):
        cls.fail_enabled = fail_enabled
        cls.call_count = 0

    def translate(self, text):
        type(self).call_count += 1
        if type(self).fail_enabled and type(self).call_count == type(self).fail_on_call:
            raise RuntimeError("forced translation failure")
        return f"T:{text}"

    def translate_list(self, plist):
        result = []
        for p in plist:
            text = p.text if hasattr(p, "text") else str(p)
            result.append(self.translate(text))
        return result


def _create_epub_with_paragraphs(path: Path):
    book = epub.EpubBook()
    book.set_identifier("resume-test")
    book.set_title("Resume Test")
    book.set_language("en")

    chapter = epub.EpubHtml(title="Chapter", file_name="chapter.xhtml", lang="en")
    chapter.content = (
        "<html><body>"
        "<p>one two three four five</p>"
        "<p>six seven eight nine ten</p>"
        "<p>eleven twelve thirteen fourteen fifteen</p>"
        "<p>sixteen seventeen eighteen nineteen twenty</p>"
        "</body></html>"
    )
    book.add_item(chapter)
    book.toc = (epub.Link("chapter.xhtml", "Chapter", "chapter"),)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]
    epub.write_epub(str(path), book)


def test_epub_resume_continues_from_checkpoint(tmp_path):
    epub_path = tmp_path / "resume_test.epub"
    _create_epub_with_paragraphs(epub_path)

    ResumeFlakyModel.reset(fail_enabled=True)
    loader = EPUBBookLoader(
        str(epub_path),
        ResumeFlakyModel,
        key="test-key",
        resume=False,
        language="zh-hans",
    )
    loader.accumulated_num = 2

    with pytest.raises(SystemExit):
        loader.make_bilingual_book()

    checkpoint_path = tmp_path / ".resume_test.temp.bin"
    assert checkpoint_path.exists()
    with open(checkpoint_path, "rb") as f:
        saved_translations = pickle.load(f)
    assert len(saved_translations) >= 2

    ResumeFlakyModel.reset(fail_enabled=False)
    loader_resume = EPUBBookLoader(
        str(epub_path),
        ResumeFlakyModel,
        key="test-key",
        resume=True,
        language="zh-hans",
    )
    loader_resume.accumulated_num = 2
    loader_resume.make_bilingual_book()

    assert ResumeFlakyModel.call_count <= 2
    output_path = tmp_path / "resume_test_bilingual.epub"
    assert output_path.exists()
    assert output_path.stat().st_size > 0
