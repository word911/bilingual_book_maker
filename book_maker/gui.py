import argparse
import json
import os
import re
import subprocess
import threading
import time
import traceback
import webbrowser
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import requests

from book_maker import cli


def _safe_int(value, default):
    try:
        return int(value)
    except Exception:
        return default


def guess_output_path(source_path):
    source = Path(source_path)
    ext = source.suffix.lower()
    if ext == ".epub":
        return str(source.with_name(f"{source.stem}_bilingual.epub"))
    if ext == ".txt":
        return str(source.with_name(f"{source.stem}_bilingual.txt"))
    if ext == ".pdf":
        return str(source.with_name(f"{source.stem}_bilingual.epub"))
    if ext == ".srt":
        return str(source.with_name(f"{source.stem}_bilingual.srt"))
    return ""


def build_models_endpoint(api_base):
    base = str(api_base or "").strip().rstrip("/")
    if not base:
        raise ValueError("api_base is required")

    lowered = base.lower()
    if lowered.endswith("/models"):
        return base
    if lowered.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


def extract_model_ids(payload):
    items = []
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            items = payload.get("data", [])
        elif isinstance(payload.get("models"), list):
            items = payload.get("models", [])
    elif isinstance(payload, list):
        items = payload

    model_ids = []
    for item in items:
        model_id = None
        if isinstance(item, str):
            model_id = item
        elif isinstance(item, dict):
            model_id = item.get("id") or item.get("model") or item.get("name")
        if isinstance(model_id, str):
            model_id = model_id.strip()
            if model_id:
                model_ids.append(model_id)

    return sorted(set(model_ids), key=lambda x: x.lower())


def discover_models(api_base, api_key="", request_get=None):
    endpoint = build_models_endpoint(api_base)
    headers = {"Accept": "application/json"}
    if str(api_key or "").strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    getter = request_get or requests.get
    response = getter(endpoint, headers=headers, timeout=15)
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code} when requesting {endpoint}")

    try:
        payload = response.json()
    except Exception as err:
        raise RuntimeError("model discovery returned non-JSON response") from err

    models = extract_model_ids(payload)
    if not models:
        raise RuntimeError("no models found in model discovery response")

    return endpoint, models


def build_cli_args(config):
    source_path = str(config.get("source_path", "")).strip()
    if not source_path:
        raise ValueError("source_path is required")

    model = str(config.get("model", "chatgptapi")).strip() or "chatgptapi"
    target_lang = str(config.get("target_lang", "zh-hans")).strip() or "zh-hans"

    args = [
        "--book_name",
        source_path,
        "--model",
        model,
        "--language",
        target_lang,
        "--no-tui",
    ]

    api_key = str(config.get("api_key", "")).strip()
    if api_key:
        args.extend(["--openai_key", api_key])

    api_base = str(config.get("api_base", "")).strip()
    if api_base:
        args.extend(["--api_base", api_base])

    if config.get("resume", True):
        args.append("--resume")

    if config.get("context_summary", False):
        args.append("--use_context")

    parallel_workers = max(1, _safe_int(config.get("parallel_workers", 1), 1))
    args.extend(["--parallel-workers", str(parallel_workers)])

    if config.get("accumulated_mode", True):
        accumulated_num = max(2, _safe_int(config.get("accumulated_num", 1200), 1200))
        args.extend(["--accumulated_num", str(accumulated_num)])
    else:
        args.extend(["--accumulated_num", "1"])

    if config.get("test_mode", False):
        test_num = max(1, _safe_int(config.get("test_num", 10), 10))
        args.extend(["--test", "--test_num", str(test_num)])

    return args


@dataclass
class SessionState:
    status: str = "idle"
    status_text: str = ""
    current_step: str = "waiting"
    source_path: str = ""
    checkpoint_path: str = ""
    output_path: str = ""
    chapter_done: int = 0
    chapter_total: int = 0
    segment_done: int = 0
    segment_total: int = 0
    started_at: float = 0.0
    ended_at: float = 0.0
    logs: deque = field(default_factory=lambda: deque(maxlen=200))
    has_warning: bool = False


class LineBufferTee:
    def __init__(self, callback: Callable[[str], None], target):
        self._callback = callback
        self._target = target
        self._buffer = ""

    def write(self, text):
        if not isinstance(text, str):
            text = str(text)
        if self._target is not None:
            self._target.write(text)
            self._target.flush()
        if not text:
            return 0
        self._buffer += text.replace("\r", "\n")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            clean = line.strip()
            if clean:
                self._callback(clean)
        return len(text)

    def flush(self):
        if self._target is not None:
            self._target.flush()
        clean = self._buffer.strip()
        if clean:
            self._callback(clean)
        self._buffer = ""


class GuiService:
    _run_summary_re = re.compile(r"\[RUN\] Chapters:\s*(\d+);\s*Target segments:\s*(\d+)")
    _chapter_re = re.compile(r"\[CHAPTER\s+(\d+)/(\d+)\]")
    _chapter_step_re = re.compile(r"\[CHAPTER\s+(\d+)/(\d+)\]\s*([^|]+)")
    _segment_re = re.compile(r"(\d+)\s*/\s*(\d+)\s*seg\b", re.IGNORECASE)
    _chapter_pbar_re = re.compile(r"(\d+)\s*/\s*(\d+)\s*ch\b")
    _ratio_re = re.compile(r"(\d+)\s*/\s*(\d+)")
    _checkpoint_re = re.compile(r"Checkpoint (?:path|saved):\s*(.+)$")
    _done_output_re = re.compile(r"\[DONE\].*Output file:\s*([^\.]+\.[a-zA-Z0-9]+)")
    _ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

    def __init__(self):
        self._lock = threading.Lock()
        self._state = SessionState()
        self._loader = None
        self._worker = None
        self._file_picker_backoff_until = 0.0
        self._file_picker_last_error = ""
        self._file_picker_fail_count = 0

    def _normalize_line(self, line):
        clean = self._ansi_re.sub("", str(line)).strip()
        return clean

    def _should_store_log(self, line):
        lower = line.lower()
        if line.startswith("Segments:") or line.startswith("Chapters:"):
            return False
        if line.startswith(
            (
                "[RUN]",
                "[CHAPTER",
                "[WARN]",
                "[ADAPTIVE]",
                "[DONE]",
                "[TUI]",
                "[GUI]",
            )
        ):
            return True
        if "translation exited with status" in lower:
            return True
        if lower.startswith("error:") or "traceback" in lower:
            return True
        return False

    def _append_log(self, line):
        clean = self._normalize_line(line)
        if not clean:
            return
        with self._lock:
            self._update_progress_from_line(clean)
            if self._should_store_log(clean):
                if not self._state.logs or self._state.logs[-1] != clean:
                    self._state.logs.append(clean)

    def _update_progress_from_line(self, line):
        summary_match = self._run_summary_re.search(line)
        if summary_match:
            self._state.chapter_total = int(summary_match.group(1))
            self._state.segment_total = int(summary_match.group(2))

        if line.startswith("Segments:"):
            ratio_match = self._ratio_re.search(line)
            if ratio_match:
                self._state.segment_done = int(ratio_match.group(1))
                self._state.segment_total = int(ratio_match.group(2))

        if line.startswith("Chapters:"):
            ratio_match = self._ratio_re.search(line)
            if ratio_match:
                self._state.chapter_done = int(ratio_match.group(1))
                self._state.chapter_total = int(ratio_match.group(2))

        chapter_match = self._chapter_re.search(line)
        if chapter_match:
            self._state.chapter_done = int(chapter_match.group(1))
            self._state.chapter_total = int(chapter_match.group(2))
            step_match = self._chapter_step_re.search(line)
            if step_match:
                raw_step = Path(step_match.group(3).strip()).name
                self._state.current_step = raw_step
            else:
                self._state.current_step = line.split("]", 1)[-1].strip()

        segment_match = self._segment_re.search(line)
        if segment_match:
            self._state.segment_done = int(segment_match.group(1))
            self._state.segment_total = int(segment_match.group(2))

        chapter_pbar_match = self._chapter_pbar_re.search(line)
        if chapter_pbar_match:
            self._state.chapter_done = int(chapter_pbar_match.group(1))
            self._state.chapter_total = int(chapter_pbar_match.group(2))

        checkpoint_match = self._checkpoint_re.search(line)
        if checkpoint_match:
            self._state.checkpoint_path = checkpoint_match.group(1).strip()

        done_match = self._done_output_re.search(line)
        if done_match:
            self._state.output_path = done_match.group(1).strip()
            self._state.current_step = "completed"

        if "[WARN]" in line or line.lower().startswith("error:"):
            self._state.has_warning = True

    def _set_loader(self, loader):
        with self._lock:
            self._loader = loader
            self._state.checkpoint_path = getattr(loader, "bin_path", "") or ""
            if not self._state.output_path and self._state.source_path:
                self._state.output_path = guess_output_path(self._state.source_path)

    def notify(self, message):
        self._append_log(message)

    def set_selected_source(self, source_path):
        source = str(source_path or "").strip()
        if not source:
            return
        with self._lock:
            self._state.source_path = source
            # Keep checkpoint/output hints in sync before a new run starts.
            source_obj = Path(source)
            self._state.checkpoint_path = str(
                source_obj.parent / f".{source_obj.stem}.temp.bin"
            )
            self._state.output_path = guess_output_path(source)

    def start(self, config):
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("translation task is already running")

            args = build_cli_args(config)
            source_path = str(config.get("source_path", "")).strip()
            self._state = SessionState(
                status="running",
                current_step="starting",
                source_path=source_path,
                checkpoint_path=str(Path(source_path).parent / f".{Path(source_path).stem}.temp.bin"),
                output_path=guess_output_path(source_path),
                started_at=time.time(),
                logs=deque(["[GUI] Translation task started."], maxlen=200),
            )
            self._loader = None

        def _run():
            error_text = ""
            with self._lock:
                self._state.status = "running"
            cli.set_loader_created_hook(self._set_loader)
            out_tee = LineBufferTee(self._append_log, None)
            err_tee = LineBufferTee(self._append_log, None)
            try:
                with redirect_stdout(out_tee), redirect_stderr(err_tee):
                    cli.main(args)
            except SystemExit as err:
                code = err.code if isinstance(err.code, int) else 0
                if code not in (0, None):
                    error_text = f"translation exited with status {code}"
                    self._append_log(f"[GUI] {error_text}")
            except Exception:
                error_text = traceback.format_exc(limit=5)
                self._append_log(error_text.strip())
            finally:
                cli.set_loader_created_hook(None)
                out_tee.flush()
                err_tee.flush()
                with self._lock:
                    self._state.ended_at = time.time()
                    if self._state.status == "stopping":
                        self._state.status = "stopped"
                        self._state.status_text = "Checkpoint saved. You can resume later."
                    elif error_text:
                        self._state.status = "error"
                        self._state.status_text = "Task failed. Check signal board."
                    else:
                        self._state.status = "completed"
                        self._state.status_text = "Translation completed."
                        if self._state.segment_total > 0:
                            self._state.segment_done = self._state.segment_total
                        if self._state.chapter_total > 0:
                            self._state.chapter_done = self._state.chapter_total
                    self._loader = None

        self._worker = threading.Thread(target=_run, daemon=True, name="bbm-gui-worker")
        self._worker.start()

    def pause(self):
        with self._lock:
            runtime_control = getattr(self._loader, "runtime_control", None)
            if runtime_control is None:
                raise RuntimeError("no active runtime control")
            runtime_control.request_pause()
            self._state.status = "pausing"
            self._state.status_text = (
                "Pause requested. Waiting current request to finish."
            )
            self._state.logs.append(
                "[GUI] Pause requested. The task will pause after the current request."
            )

    def resume(self):
        with self._lock:
            runtime_control = getattr(self._loader, "runtime_control", None)
            if runtime_control is None:
                raise RuntimeError("no active runtime control")
            runtime_control.request_resume()
            self._state.status = "running"
            self._state.status_text = "Resumed from GUI."
            self._state.logs.append("[GUI] Resume requested.")

    def request_exit(self):
        with self._lock:
            runtime_control = getattr(self._loader, "runtime_control", None)
            if runtime_control is None:
                raise RuntimeError("no active runtime control")
            runtime_control.request_exit()
            self._state.status = "stopping"
            self._state.status_text = (
                "Save-and-exit requested. Waiting current request, then checkpoint."
            )
            checkpoint_hint = self._state.checkpoint_path or "<unknown>"
            self._state.logs.append(
                "[GUI] Save-and-exit requested. "
                f"Will save checkpoint to {checkpoint_hint} and stop."
            )

    @staticmethod
    def _pick_file_windows_powershell():
        script = r"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Filter = "Supported files (*.epub;*.txt;*.pdf;*.srt)|*.epub;*.txt;*.pdf;*.srt|EPUB (*.epub)|*.epub|TXT (*.txt)|*.txt|PDF (*.pdf)|*.pdf|SRT (*.srt)|*.srt|All files (*.*)|*.*"
$dialog.Multiselect = $false
$dialog.CheckFileExists = $true
$result = $dialog.ShowDialog()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $dialog.FileName }
"""
        try:
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-STA", "-Command", script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=20,
            )
        except subprocess.TimeoutExpired as err:
            raise RuntimeError("powershell file dialog timed out") from err
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            raise RuntimeError(stderr or f"powershell exit code {completed.returncode}")
        return (completed.stdout or "").strip()

    @staticmethod
    def _pick_file_tkinter():
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            filetypes=[
                ("Supported files", "*.epub *.txt *.pdf *.srt"),
                ("EPUB", "*.epub"),
                ("TXT", "*.txt"),
                ("PDF", "*.pdf"),
                ("SRT", "*.srt"),
                ("All files", "*.*"),
            ]
        )
        root.destroy()
        return path or ""

    def pick_file(self):
        now = time.time()
        with self._lock:
            if now < self._file_picker_backoff_until:
                remain = int(max(1, self._file_picker_backoff_until - now))
                reason = self._file_picker_last_error or "file dialog unavailable"
                return "", f"{reason} (retry in {remain}s)"

        errors = []
        if os.name == "nt":
            try:
                path = self._pick_file_windows_powershell()
                with self._lock:
                    self._file_picker_fail_count = 0
                    self._file_picker_backoff_until = 0.0
                    self._file_picker_last_error = ""
                return path, ""
            except Exception as err:
                errors.append(f"powershell dialog failed: {err}")

        try:
            path = self._pick_file_tkinter()
            with self._lock:
                self._file_picker_fail_count = 0
                self._file_picker_backoff_until = 0.0
                self._file_picker_last_error = ""
            return path, ""
        except Exception as err:
            errors.append(f"tkinter dialog failed: {err}")

        error_text = " | ".join(errors) or "file dialog unavailable"
        with self._lock:
            self._file_picker_fail_count += 1
            cooldown = min(90, 10 * self._file_picker_fail_count)
            self._file_picker_backoff_until = time.time() + cooldown
            self._file_picker_last_error = error_text
        return "", error_text

    def state_snapshot(self):
        with self._lock:
            state = self._state
            runtime_control = getattr(self._loader, "runtime_control", None)
            if runtime_control is not None and state.status in (
                "running",
                "pausing",
                "paused",
            ):
                if runtime_control.paused:
                    state.status = "paused"
                    if not state.status_text or "Pause requested" in state.status_text:
                        state.status_text = "Paused."
                elif state.status == "paused":
                    state.status = "running"
                    state.status_text = "Running."

            if state.started_at <= 0:
                runtime_seconds = 0
            elif state.ended_at > 0:
                runtime_seconds = int(max(0, state.ended_at - state.started_at))
            else:
                runtime_seconds = int(max(0, time.time() - state.started_at))

            return {
                "status": state.status,
                "status_text": state.status_text,
                "current_step": state.current_step,
                "source_path": state.source_path,
                "checkpoint_path": state.checkpoint_path,
                "output_path": state.output_path,
                "chapter_done": state.chapter_done,
                "chapter_total": state.chapter_total,
                "segment_done": state.segment_done,
                "segment_total": state.segment_total,
                "runtime_seconds": runtime_seconds,
                "logs": list(state.logs),
                "has_warning": state.has_warning,
            }


class GuiRequestHandler(BaseHTTPRequestHandler):
    service = GuiService()

    @staticmethod
    def _is_client_disconnect_error(err):
        if isinstance(err, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return True
        if isinstance(err, OSError):
            return getattr(err, "winerror", None) in (10053, 10054, 10058)
        return False

    def _send_json(self, status_code, payload):
        payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload_bytes)))
            self.end_headers()
            self.wfile.write(payload_bytes)
            return True
        except Exception as err:
            if self._is_client_disconnect_error(err):
                return False
            raise

    def _read_json(self):
        content_length = _safe_int(self.headers.get("Content-Length", "0"), 0)
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        try:
            if self.path == "/api/state":
                self._send_json(200, self.service.state_snapshot())
                return

            if self.path == "/" or self.path.startswith("/?"):
                candidate_paths = [
                    Path(__file__).with_name("gui_assets") / "index.html",
                    Path(__file__).resolve().parents[1] / "docs" / "gui_memphis_prototype.html",
                ]
                html_path = next((path for path in candidate_paths if path.exists()), None)
                if html_path is None:
                    self._send_json(500, {"error": "GUI asset missing"})
                    return
                html = html_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return

            self._send_json(404, {"error": "not found"})
        except Exception as err:
            if self._is_client_disconnect_error(err):
                return
            raise

    def do_POST(self):
        try:
            if self.path == "/api/start":
                payload = self._read_json()
                self.service.start(payload)
                self._send_json(200, {"ok": True})
                return

            if self.path == "/api/pause":
                self.service.pause()
                self._send_json(200, {"ok": True})
                return

            if self.path == "/api/resume":
                self.service.resume()
                self._send_json(200, {"ok": True})
                return

            if self.path == "/api/exit":
                self.service.request_exit()
                self._send_json(200, {"ok": True})
                return

            if self.path == "/api/pick-file":
                self.service.notify("[GUI] Opening file picker...")
                path, error = self.service.pick_file()
                if path:
                    self.service.set_selected_source(path)
                    self.service.notify(f"[GUI] Selected source file: {path}")
                elif error:
                    self.service.notify(f"[WARN] Browse unavailable: {error}")
                else:
                    self.service.notify("[GUI] File picker canceled.")
                self._send_json(200, {"ok": bool(path), "path": path, "error": error})
                return

            if self.path == "/api/models":
                payload = self._read_json()
                api_base = str(payload.get("api_base", "")).strip()
                api_key = str(payload.get("api_key", "")).strip()
                try:
                    endpoint, models = discover_models(api_base=api_base, api_key=api_key)
                    self.service.notify(
                        f"[RUN] Model discovery: {len(models)} models from {endpoint}"
                    )
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "models": models,
                            "endpoint": endpoint,
                            "error": "",
                        },
                    )
                except Exception as err:
                    self.service.notify(f"[WARN] Model discovery failed: {err}")
                    self._send_json(
                        200,
                        {
                            "ok": False,
                            "models": [],
                            "endpoint": "",
                            "error": str(err),
                        },
                    )
                return

            self._send_json(404, {"error": "not found"})
        except Exception as err:
            if self._is_client_disconnect_error(err):
                return
            try:
                self._send_json(400, {"error": str(err)})
            except Exception as send_err:
                if self._is_client_disconnect_error(send_err):
                    return
                raise


def run_server(host="127.0.0.1", port=8765, open_browser_flag=True):
    server = ThreadingHTTPServer((host, port), GuiRequestHandler)
    url = f"http://{host}:{port}"
    print(f"[GUI] Serving at {url}")
    if open_browser_flag:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Launch bbook_maker Web GUI")
    parser.add_argument("--host", default="127.0.0.1", help="bind host")
    parser.add_argument("--port", type=int, default=8765, help="bind port")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open browser automatically",
    )
    args = parser.parse_args(argv)
    run_server(args.host, args.port, open_browser_flag=not args.no_browser)


if __name__ == "__main__":
    main()
