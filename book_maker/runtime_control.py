import os
import sys
import time
from threading import Event, Lock, Thread

from tqdm import tqdm


class RuntimeControl:
    """Terminal runtime controller for pause/resume/checkpoint-exit.

    Hotkeys:
    - p: pause
    - c: continue
    - q: save checkpoint and exit (handled cooperatively by loaders)
    """

    def __init__(self, checkpoint_path=None, enabled=True):
        self.checkpoint_path = checkpoint_path
        self.enabled = bool(enabled and sys.stdin.isatty() and sys.stdout.isatty())
        self._paused = Event()
        self._exit_requested = Event()
        self._stop_requested = Event()
        self._listener_thread = None
        self._log_lock = Lock()
        self._pause_banner_printed = False

    @staticmethod
    def _ui_log(message):
        try:
            tqdm.write(message)
        except Exception:
            print(message)

    def start(self):
        if not self.enabled:
            return
        self._ui_log(
            "[TUI] Controls: [p] pause, [c] continue, [q] save checkpoint and exit."
        )
        self._listener_thread = Thread(
            target=self._listen_keyboard,
            name="bbm-runtime-control",
            daemon=True,
        )
        self._listener_thread.start()

    def stop(self):
        self._stop_requested.set()
        if self._listener_thread is not None and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=1.0)

    @property
    def paused(self):
        return self._paused.is_set()

    @property
    def exit_requested(self):
        return self._exit_requested.is_set()

    def request_pause(self):
        if self._paused.is_set():
            return
        self._paused.set()
        checkpoint_hint = self.checkpoint_path or "<unknown>"
        self._ui_log(
            "[TUI] Paused. Press [c] to continue, [q] to save checkpoint and exit. "
            f"Checkpoint path: {checkpoint_hint}"
        )

    def request_resume(self):
        if not self._paused.is_set():
            return
        self._paused.clear()
        self._pause_banner_printed = False
        self._ui_log("[TUI] Resumed.")

    def request_exit(self):
        if self._exit_requested.is_set():
            return
        self._exit_requested.set()
        checkpoint_hint = self.checkpoint_path or "<unknown>"
        self._ui_log(
            "[TUI] Exit requested. The task will save checkpoint and quit soon. "
            f"Checkpoint path: {checkpoint_hint}"
        )

    def wait_if_paused(self):
        while self._paused.is_set() and not self._exit_requested.is_set():
            if not self._pause_banner_printed:
                self._ui_log("[TUI] Waiting in paused state...")
                self._pause_banner_printed = True
            time.sleep(0.2)

    def _listen_keyboard(self):
        if os.name == "nt":
            self._listen_keyboard_windows()
        else:
            self._listen_keyboard_posix()

    def _listen_keyboard_windows(self):
        import msvcrt

        while not self._stop_requested.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                self._handle_key(ch)
            time.sleep(0.05)

    def _listen_keyboard_posix(self):
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop_requested.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    ch = sys.stdin.read(1)
                    self._handle_key(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _handle_key(self, key):
        if not key:
            return
        key = key.lower()
        with self._log_lock:
            if key == "p":
                self.request_pause()
            elif key == "c":
                self.request_resume()
            elif key == "q":
                self.request_exit()
