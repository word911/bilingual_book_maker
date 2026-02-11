import sys
import types
import importlib.util

import pytest

if importlib.util.find_spec("ebooklib") is None:
    ebooklib_stub = types.ModuleType("ebooklib")
    ebooklib_stub.ITEM_DOCUMENT = 0
    ebooklib_stub.epub = types.ModuleType("ebooklib.epub")
    sys.modules["ebooklib"] = ebooklib_stub
    sys.modules["ebooklib.epub"] = ebooklib_stub.epub

from book_maker import cli


class DummyTranslator:
    instances = []

    def __init__(self, key, language, api_base=None, **kwargs):
        self.key = key
        self.language = language
        self.api_base = api_base
        self.model_list = None
        self.rpm = None
        DummyTranslator.instances.append(self)

    def set_model_list(self, model_list):
        self.model_list = model_list

    def set_rpm(self, rpm):
        self.rpm = rpm


class DummyLoader:
    last_instance = None

    def __init__(
        self,
        book_name,
        model,
        key,
        resume,
        language,
        model_api_base=None,
        **kwargs,
    ):
        self.book_name = book_name
        self.resume = resume
        self.token_estimator_model = "gpt-3.5-turbo-0301"
        self.accumulated_min_num = None
        self.accumulated_backoff_factor = None
        self.accumulated_recover_factor = None
        self.accumulated_recover_successes = None
        self.runtime_control_calls = []
        self.translate_model = model(key, language, api_base=model_api_base)
        DummyLoader.last_instance = self

    def setup_runtime_control(self, tui_enabled=True):
        self.runtime_control_calls.append(tui_enabled)

    def make_bilingual_book(self):
        return


@pytest.fixture(autouse=True)
def reset_state():
    DummyTranslator.instances.clear()
    DummyLoader.last_instance = None


def test_cli_accepts_openai_compatible_custom_model(monkeypatch, tmp_path):
    test_book = tmp_path / "demo.txt"
    test_book.write_text("hello", encoding="utf-8")

    monkeypatch.setattr(cli, "MODEL_DICT", {"openai": DummyTranslator})
    monkeypatch.setattr(cli, "BOOK_LOADER_DICT", {"txt": DummyLoader})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bbook_maker",
            "--book_name",
            str(test_book),
            "--openai_key",
            "test-key",
            "--api_base",
            "https://example.com/v1",
            "--model",
            "DeepSeek-V3.2",
            "--rpm",
            "30",
        ],
    )

    cli.main()

    assert DummyLoader.last_instance is not None
    assert DummyLoader.last_instance.translate_model.api_base == "https://example.com/v1"
    assert DummyLoader.last_instance.translate_model.model_list == ["DeepSeek-V3.2"]
    assert DummyLoader.last_instance.translate_model.rpm == 30.0
    assert DummyLoader.last_instance.token_estimator_model == "DeepSeek-V3.2"


def test_cli_rejects_unknown_model_without_api_base(monkeypatch, tmp_path, capsys):
    test_book = tmp_path / "demo.txt"
    test_book.write_text("hello", encoding="utf-8")

    monkeypatch.setattr(cli, "MODEL_DICT", {"openai": DummyTranslator})
    monkeypatch.setattr(cli, "BOOK_LOADER_DICT", {"txt": DummyLoader})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bbook_maker",
            "--book_name",
            str(test_book),
            "--openai_key",
            "test-key",
            "--model",
            "DeepSeek-V3.2",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert "unsupported model" in capsys.readouterr().err


def test_cli_rejects_negative_rpm(monkeypatch, tmp_path, capsys):
    test_book = tmp_path / "demo.txt"
    test_book.write_text("hello", encoding="utf-8")

    monkeypatch.setattr(cli, "MODEL_DICT", {"openai": DummyTranslator})
    monkeypatch.setattr(cli, "BOOK_LOADER_DICT", {"txt": DummyLoader})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bbook_maker",
            "--book_name",
            str(test_book),
            "--openai_key",
            "test-key",
            "--model",
            "openai",
            "--model_list",
            "gpt-4o-mini",
            "--rpm",
            "-1",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert "`--rpm` must be >= 0" in capsys.readouterr().err


def test_cli_auto_resume_when_checkpoint_exists(monkeypatch, tmp_path):
    test_book = tmp_path / "demo.txt"
    test_book.write_text("hello", encoding="utf-8")
    checkpoint_file = tmp_path / ".demo.temp.bin"
    checkpoint_file.write_bytes(b"checkpoint")

    monkeypatch.setattr(cli, "MODEL_DICT", {"openai": DummyTranslator})
    monkeypatch.setattr(cli, "BOOK_LOADER_DICT", {"txt": DummyLoader})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bbook_maker",
            "--book_name",
            str(test_book),
            "--openai_key",
            "test-key",
            "--model",
            "openai",
            "--model_list",
            "gpt-4o-mini",
        ],
    )

    cli.main()

    assert DummyLoader.last_instance is not None
    assert DummyLoader.last_instance.resume is True
    assert DummyLoader.last_instance.token_estimator_model == "gpt-4o-mini"


def test_cli_sets_accumulated_adaptive_options(monkeypatch, tmp_path):
    test_book = tmp_path / "demo.txt"
    test_book.write_text("hello", encoding="utf-8")

    monkeypatch.setattr(cli, "MODEL_DICT", {"openai": DummyTranslator})
    monkeypatch.setattr(cli, "BOOK_LOADER_DICT", {"txt": DummyLoader})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bbook_maker",
            "--book_name",
            str(test_book),
            "--openai_key",
            "test-key",
            "--model",
            "openai",
            "--model_list",
            "gpt-4o-mini",
            "--accumulated_num",
            "1200",
            "--accumulated-min-num",
            "300",
            "--accumulated-backoff-factor",
            "0.6",
            "--accumulated-recover-factor",
            "1.25",
            "--accumulated-recover-successes",
            "4",
        ],
    )

    cli.main()

    assert DummyLoader.last_instance is not None
    assert DummyLoader.last_instance.accumulated_min_num == 300
    assert DummyLoader.last_instance.accumulated_backoff_factor == 0.6
    assert DummyLoader.last_instance.accumulated_recover_factor == 1.25
    assert DummyLoader.last_instance.accumulated_recover_successes == 4


def test_cli_enables_tui_by_default(monkeypatch, tmp_path):
    test_book = tmp_path / "demo.txt"
    test_book.write_text("hello", encoding="utf-8")

    monkeypatch.setattr(cli, "MODEL_DICT", {"openai": DummyTranslator})
    monkeypatch.setattr(cli, "BOOK_LOADER_DICT", {"txt": DummyLoader})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bbook_maker",
            "--book_name",
            str(test_book),
            "--openai_key",
            "test-key",
            "--model",
            "openai",
            "--model_list",
            "gpt-4o-mini",
        ],
    )

    cli.main()

    assert DummyLoader.last_instance is not None
    assert DummyLoader.last_instance.runtime_control_calls == [True]


def test_cli_disables_tui_when_no_tui_flag_is_set(monkeypatch, tmp_path):
    test_book = tmp_path / "demo.txt"
    test_book.write_text("hello", encoding="utf-8")

    monkeypatch.setattr(cli, "MODEL_DICT", {"openai": DummyTranslator})
    monkeypatch.setattr(cli, "BOOK_LOADER_DICT", {"txt": DummyLoader})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bbook_maker",
            "--book_name",
            str(test_book),
            "--openai_key",
            "test-key",
            "--model",
            "openai",
            "--model_list",
            "gpt-4o-mini",
            "--no-tui",
        ],
    )

    cli.main()

    assert DummyLoader.last_instance is not None
    assert DummyLoader.last_instance.runtime_control_calls == [False]
