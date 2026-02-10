from bs4 import BeautifulSoup as bs
import pytest

ebooklib = pytest.importorskip("ebooklib")
from ebooklib import epub

from book_maker.loader.epub_loader import EPUBBookLoader


class AdaptiveGatewayModel:
    gateway_events = []
    batch_sizes = []

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
    def reset(cls, gateway_events):
        cls.gateway_events = list(gateway_events)
        cls.batch_sizes = []

    def translate(self, text):
        return f"T:{text}"

    def translate_list(self, plist):
        type(self).batch_sizes.append(len(plist))
        result = []
        for p in plist:
            text = p.text if hasattr(p, "text") else str(p)
            result.append(f"T:{text}")
        return result

    def consume_gateway_timeout_events(self):
        if type(self).gateway_events:
            return type(self).gateway_events.pop(0)
        return 0


def _create_epub(path):
    book = epub.EpubBook()
    book.set_identifier("adaptive-test")
    book.set_title("Adaptive Test")
    book.set_language("en")

    chapter = epub.EpubHtml(title="Chapter", file_name="chapter.xhtml", lang="en")
    chapter.content = "<html><body><p>placeholder</p></body></html>"
    book.add_item(chapter)
    book.toc = (epub.Link("chapter.xhtml", "Chapter", "chapter"),)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]
    epub.write_epub(str(path), book)


def test_accumulated_num_auto_backoff_on_gateway_events(tmp_path):
    epub_path = tmp_path / "adaptive_test.epub"
    _create_epub(epub_path)
    AdaptiveGatewayModel.reset(gateway_events=[1, 0, 0, 0, 0])

    loader = EPUBBookLoader(
        str(epub_path),
        AdaptiveGatewayModel,
        key="test-key",
        resume=False,
        language="zh-hans",
    )
    loader.accumulated_num = 120
    loader.accumulated_min_num = 60
    loader.accumulated_backoff_factor = 0.5
    loader.accumulated_recover_successes = 100
    loader.accumulated_recover_factor = 1.2
    loader._dynamic_accumulated_num = None
    loader._estimate_tokens = lambda _: 29

    soup = bs(
        "<p>a</p><p>b</p><p>c</p><p>d</p><p>e</p><p>f</p><p>g</p><p>h</p>",
        "html.parser",
    )
    p_list = soup.find_all("p")

    index = loader.translate_paragraphs_acc(
        p_list,
        send_num=loader.accumulated_num,
        index=0,
        p_to_save_len=0,
    )

    assert index == 8
    assert AdaptiveGatewayModel.batch_sizes[0] == 4
    assert 2 in AdaptiveGatewayModel.batch_sizes[1:]
    assert loader._dynamic_accumulated_num == 60
