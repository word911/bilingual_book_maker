from pathlib import Path

from book_maker.gui import GuiService, build_cli_args, guess_output_path
import book_maker.gui as gui_module


def test_guess_output_path_for_supported_types(tmp_path):
    epub = tmp_path / "book.epub"
    txt = tmp_path / "book.txt"
    pdf = tmp_path / "book.pdf"
    srt = tmp_path / "book.srt"

    assert guess_output_path(str(epub)).endswith("book_bilingual.epub")
    assert guess_output_path(str(txt)).endswith("book_bilingual.txt")
    assert guess_output_path(str(pdf)).endswith("book_bilingual.epub")
    assert guess_output_path(str(srt)).endswith("book_bilingual.srt")


def test_build_cli_args_with_common_gui_options():
    args = build_cli_args(
        {
            "source_path": "E:/books/demo.epub",
            "model": "gpt-5-nano",
            "target_lang": "zh-hans",
            "api_base": "https://gateway.example.com/v1",
            "api_key": "sk-demo",
            "resume": True,
            "context_summary": True,
            "accumulated_mode": True,
            "accumulated_num": 900,
            "parallel_workers": 3,
        }
    )

    assert "--book_name" in args
    assert "E:/books/demo.epub" in args
    assert "--model" in args
    assert "gpt-5-nano" in args
    assert "--api_base" in args
    assert "https://gateway.example.com/v1" in args
    assert "--openai_key" in args
    assert "sk-demo" in args
    assert "--resume" in args
    assert "--use_context" in args
    assert "--accumulated_num" in args
    assert "900" in args
    assert "--parallel-workers" in args
    assert "3" in args
    assert "--rpm" in args
    rpm_index = args.index("--rpm")
    assert args[rpm_index + 1] == "8"
    assert "--no-tui" in args


def test_build_cli_args_can_disable_accumulated_mode():
    args = build_cli_args(
        {
            "source_path": str(Path("demo.txt")),
            "accumulated_mode": False,
            "resume": False,
        }
    )

    assert "--accumulated_num" in args
    index = args.index("--accumulated_num")
    assert args[index + 1] == "1"
    assert "--resume" not in args


def test_build_cli_args_caps_rpm_below_ten():
    args = build_cli_args(
        {
            "source_path": "E:/books/demo.epub",
            "rpm": 20,
        }
    )

    assert "--rpm" in args
    rpm_index = args.index("--rpm")
    assert args[rpm_index + 1] == "9.9"


def test_gui_service_parses_segment_tqdm_ratio_without_log_noise():
    service = GuiService()
    service._append_log(
        "Segments: 93%|##########| 5517/5935 [00:03<00:00, 1946.66seg/s]"
    )

    snapshot = service.state_snapshot()
    assert snapshot["segment_done"] == 5517
    assert snapshot["segment_total"] == 5935
    assert snapshot["logs"] == []


def test_gui_service_keeps_key_chapter_log_and_updates_step():
    service = GuiService()
    line = "[CHAPTER 34/37] Autopilot_ebook-32.xhtml | segments=28 | mode=accumulated"
    service._append_log(line)

    snapshot = service.state_snapshot()
    assert snapshot["chapter_done"] == 34
    assert snapshot["chapter_total"] == 37
    assert snapshot["current_step"] == "Autopilot_ebook-32.xhtml"
    assert snapshot["logs"][-1] == line


def test_pick_file_backoff_avoids_repeated_blocking(monkeypatch):
    service = GuiService()
    calls = {"ps": 0, "tk": 0}

    def _fail_ps():
        calls["ps"] += 1
        raise RuntimeError("powershell file dialog timed out")

    def _fail_tk():
        calls["tk"] += 1
        raise RuntimeError("No module named 'tkinter'")

    monkeypatch.setattr(gui_module.os, "name", "nt", raising=False)
    monkeypatch.setattr(service, "_pick_file_windows_powershell", _fail_ps)
    monkeypatch.setattr(service, "_pick_file_tkinter", _fail_tk)

    _path1, err1 = service.pick_file()
    assert "powershell dialog failed" in err1
    assert calls["ps"] == 1
    assert calls["tk"] == 1

    _path2, err2 = service.pick_file()
    assert "retry in" in err2
    assert calls["ps"] == 1
    assert calls["tk"] == 1
