import threading
import time

from book_maker.runtime_control import RuntimeControl


def test_runtime_control_pause_resume_and_exit_flags():
    control = RuntimeControl(checkpoint_path="dummy.bin", enabled=False)

    assert control.paused is False
    assert control.exit_requested is False

    control.request_pause()
    assert control.paused is True

    control.request_resume()
    assert control.paused is False

    control.request_exit()
    assert control.exit_requested is True


def test_runtime_control_wait_if_paused_unblocks_on_resume():
    control = RuntimeControl(checkpoint_path="dummy.bin", enabled=False)
    control.request_pause()

    finished = {"value": False}

    def _waiter():
        control.wait_if_paused()
        finished["value"] = True

    t = threading.Thread(target=_waiter)
    t.start()
    time.sleep(0.2)
    assert finished["value"] is False
    control.request_resume()
    t.join(timeout=2.0)
    assert finished["value"] is True
