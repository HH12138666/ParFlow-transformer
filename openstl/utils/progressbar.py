"""Minimal progress reporting used by evaluation loops."""

import sys
from shutil import get_terminal_size
from time import monotonic


class Timer:
    def __init__(self):
        self.started_at = monotonic()

    def since_start(self):
        return monotonic() - self.started_at


class ProgressBar:
    def __init__(self, task_num=0, bar_width=50, file=sys.stdout):
        self.task_num = int(task_num)
        self.bar_width = int(bar_width)
        self.completed = 0
        self.file = file
        self.timer = Timer()
        self._render()

    def update(self, num_tasks=1):
        if num_tasks <= 0:
            raise ValueError("num_tasks must be positive")
        self.completed += num_tasks
        self._render()

    def _render(self):
        elapsed = self.timer.since_start()
        rate = self.completed / elapsed if elapsed > 0 else 0.0
        if self.task_num <= 0:
            message = f"\rcompleted: {self.completed}, elapsed: {elapsed:.0f}s, {rate:.1f} task/s"
        else:
            message = self._task_message(elapsed, rate)
        self.file.write(message)
        self.file.flush()

    def _task_message(self, elapsed, rate):
        progress = min(self.completed / self.task_num, 1.0)
        eta = elapsed * (1.0 - progress) / progress if progress > 0 else 0.0
        suffix = (
            f" {self.completed}/{self.task_num}, {rate:.1f} task/s, "
            f"elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s"
        )
        terminal_width = get_terminal_size(fallback=(100, 20)).columns
        width = max(2, min(self.bar_width, terminal_width - len(suffix) - 3))
        marks = int(width * progress)
        return f"\r[{'>' * marks}{' ' * (width - marks)}]{suffix}"
