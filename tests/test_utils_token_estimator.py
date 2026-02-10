from book_maker.utils import num_tokens_from_text


def test_num_tokens_from_text_supports_unknown_model_name():
    tokens = num_tokens_from_text("hello world", model="DeepSeek-V3.2")
    assert isinstance(tokens, int)
    assert tokens > 0


def test_num_tokens_from_text_supports_gpt5_nano_name():
    tokens = num_tokens_from_text("This is a token estimation smoke test.", model="gpt-5-nano")
    assert isinstance(tokens, int)
    assert tokens > 0
