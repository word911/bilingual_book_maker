from bs4 import BeautifulSoup as bs

from book_maker.translator.chatgptapi_translator import ChatGPTAPI


def test_translate_retries_transient_error(monkeypatch):
    translator = ChatGPTAPI("test-key", "zh-hans")
    calls = {"count": 0}

    def fake_get_translation(_text):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError("Request timed out.")
        return "ok"

    monkeypatch.setattr(translator, "get_translation", fake_get_translation)
    monkeypatch.setattr("book_maker.translator.chatgptapi_translator.time.sleep", lambda _: None)

    result = translator.translate("hello", needprint=False)

    assert result == "ok"
    assert calls["count"] == 2


def test_translate_list_fallback_to_single_translation(monkeypatch):
    translator = ChatGPTAPI("test-key", "zh-hans")

    def fake_translate(text, needprint=False):
        if "PARAGRAPH 1:" in text and "PARAGRAPH 2:" in text:
            raise TimeoutError("batch timeout")
        if "PARAGRAPH 1:" in text:
            return "UNPARSEABLE"
        return f"T:{text}"

    monkeypatch.setattr(translator, "translate", fake_translate)

    soup = bs("<p>first</p><p>second</p>", "html.parser")
    plist = soup.find_all("p")

    results = translator.translate_list(plist)

    assert results == ["T:first", "T:second"]


def test_translate_gateway_timeout_triggers_cooldown(monkeypatch):
    translator = ChatGPTAPI("test-key", "zh-hans")
    translator.set_gateway_cooldown(threshold=2, cooldown_seconds=5)
    calls = {"count": 0}
    sleep_calls = []

    def fake_get_translation(_text):
        calls["count"] += 1
        if calls["count"] <= 2:
            raise RuntimeError("504 Gateway time-out")
        return "ok"

    monkeypatch.setattr(translator, "get_translation", fake_get_translation)
    monkeypatch.setattr(
        "book_maker.translator.chatgptapi_translator.time.sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    result = translator.translate("hello", needprint=False)

    assert result == "ok"
    assert calls["count"] == 3
    assert 5 in sleep_calls
    assert translator.consume_gateway_timeout_events() == 2
    assert translator.consume_gateway_timeout_events() == 0


def test_translate_list_auto_split_batches(monkeypatch):
    translator = ChatGPTAPI("test-key", "zh-hans")
    split_calls = []

    def fake_translate_list_once(plist):
        split_calls.append(len(plist))
        if len(plist) > 1:
            raise RuntimeError("batch timeout")
        text = translator._extract_paragraph_text(plist[0])
        return [f"T:{text}"]

    monkeypatch.setattr(translator, "_translate_list_once", fake_translate_list_once)

    soup = bs("<p>first</p><p>second</p><p>third</p><p>fourth</p>", "html.parser")
    plist = soup.find_all("p")

    results = translator.translate_list(plist)

    assert results == ["T:first", "T:second", "T:third", "T:fourth"]
    assert split_calls[0] == 4
    assert 2 in split_calls


def test_retry_error_message_is_concise_and_contains_request_id(tmp_path):
    translator = ChatGPTAPI("test-key", "zh-hans")
    translator._provider_error_log_path = str(tmp_path / "provider_error.log")

    class FakeError(Exception):
        def __init__(self):
            self.response = type(
                "Resp",
                (),
                {"headers": {"x-request-id": "req_test_123"}, "status_code": 503},
            )()
            self.status_code = 503

        def __str__(self):
            return "Gateway timeout " + ("detail " * 80)

    err = FakeError()
    translator._log_provider_error(err)
    message = translator._format_retry_error(err, retry_seconds=5)

    assert "request_id=req_test_123" in message
    assert "status=503" in message
    assert "Service Unavailable" in message
    assert "Retry in 5s." in message
    assert "provider_error.log" in message
    assert "req_test_123" in (
        (tmp_path / "provider_error.log").read_text(encoding="utf-8")
    )


def test_translate_list_retries_missing_paragraphs_individually(monkeypatch, capsys):
    translator = ChatGPTAPI("test-key", "zh-hans")

    def fake_translate(text, needprint=False):
        if "PARAGRAPH 1:" in text and "PARAGRAPH 2:" in text:
            return "TRANSLATION OF PARAGRAPH 1:\nT:first\n\n"
        return f"T:{text}"

    monkeypatch.setattr(translator, "translate", fake_translate)

    soup = bs("<p>first</p><p>second</p>", "html.parser")
    plist = soup.find_all("p")

    results = translator.translate_list(plist)

    captured = capsys.readouterr()
    assert "[WARN] Batch parse mismatch: missing 1/2 paragraphs" in captured.out
    assert "Could not find translation for paragraph" not in captured.out
    assert results == ["T:first", "T:second"]
