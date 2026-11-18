"""Run logging: a timestamping stdout tee and a section timer."""

import time


class Tee:
    """Write to both a file and an original stream, prepending timestamps."""
    def __init__(self, file, stream):
        self.file = file
        self.stream = stream
        self._at_line_start = True
    def write(self, data):
        if not data:
            return
        from datetime import datetime
        ts = datetime.now().strftime("[%H:%M:%S] ")
        lines = data.split("\n")
        stamped = []
        for i, line in enumerate(lines):
            if i > 0:
                stamped.append("\n")
            if self._at_line_start and line:
                stamped.append(ts + line)
            else:
                stamped.append(line)
            self._at_line_start = (i < len(lines) - 1) or data.endswith("\n")
        out = "".join(stamped)
        self.stream.write(out)
        self.file.write(out)
        self.file.flush()
    def fileno(self):
        return self.stream.fileno()
    def flush(self):
        self.stream.flush()
        self.file.flush()
    def close(self):
        self.file.close()


class _Timer:
    """Context manager for timing a code block and appending to a metrics list.

    Usage:
        with self._time("timing/update"):
            ...  # timed code
    """
    def __init__(self, metrics_dict, key):
        self._list = metrics_dict.setdefault("train", {}).setdefault(key, [])
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        self._list.append(time.perf_counter() - self._t0)
        return False
