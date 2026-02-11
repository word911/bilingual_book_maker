from abc import ABC, abstractmethod

from book_maker.runtime_control import RuntimeControl


class BaseBookLoader(ABC):
    def setup_runtime_control(self, tui_enabled=True):
        checkpoint_path = getattr(self, "bin_path", None)
        self.runtime_control = RuntimeControl(
            checkpoint_path=checkpoint_path,
            enabled=tui_enabled,
        )
        self.runtime_control.start()

    def teardown_runtime_control(self):
        runtime_control = getattr(self, "runtime_control", None)
        if runtime_control is not None:
            runtime_control.stop()

    def runtime_checkpoint(self):
        runtime_control = getattr(self, "runtime_control", None)
        if runtime_control is None:
            return
        runtime_control.wait_if_paused()
        if runtime_control.exit_requested:
            raise KeyboardInterrupt("Exit requested from TUI control")

    def runtime_control_status(self):
        runtime_control = getattr(self, "runtime_control", None)
        if runtime_control is None:
            return "OFF"
        return "ON"

    def runtime_checkpoint_path(self):
        checkpoint_path = getattr(self, "bin_path", "")
        return checkpoint_path or "<unknown>"

    @staticmethod
    def _is_special_text(text):
        return text.isdigit() or text.isspace()

    @abstractmethod
    def _make_new_book(self, book):
        pass

    @abstractmethod
    def make_bilingual_book(self):
        pass

    @abstractmethod
    def load_state(self):
        pass

    @abstractmethod
    def _save_temp_book(self):
        pass

    @abstractmethod
    def _save_progress(self):
        pass
