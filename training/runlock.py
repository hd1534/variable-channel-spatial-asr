"""One writer per output directory: an exclusive lock held for a whole command.

A trainer started with ``start_new_session`` can outlive a crashed notebook
kernel.  Running the notebook again would then start a second ``train.py`` (or
``eval_t2.py``) on the same directory, and the two would prune each other's
checkpoints and mix their results.  :func:`exclusive_lock` makes the second one
stop at once instead.

The lock is a non-blocking ``flock`` on a ``*.lock`` file inside the directory
(the Drive backup skips ``*.lock`` files).  The kernel drops it when the holder
exits, however it dies, so a leftover file never blocks a later run.  Without
``fcntl`` (Windows), or on a file system that refuses ``flock``, nothing is
locked and the command runs as before.
"""

from __future__ import annotations

import contextlib
import errno
import os
import sys
from collections.abc import Iterator
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

_HELD_ELSEWHERE: frozenset[int] = frozenset({errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES})


def _holder(fd: int) -> str:
    """The process id the current holder wrote into the lock file, or ``"unknown"``."""
    try:
        return os.pread(fd, 32, 0).decode("ascii", "replace").strip() or "unknown"
    except OSError:
        return "unknown"


def _acquire(fd: int, path: Path, busy: str) -> bool:
    """Take the lock on ``fd``; False when the file system cannot lock at all.

    Raises ``SystemExit(busy)`` when another process holds it.
    """
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in _HELD_ELSEWHERE:
            raise SystemExit(busy.replace("{pid}", _holder(fd))) from None
        print(f"[lock] {path}: file locking is not supported here ({exc.strerror}); continuing without a lock",
              file=sys.stderr, flush=True)
        return False
    return True


@contextlib.contextmanager
def exclusive_lock(path: str | Path, busy: str) -> Iterator[None]:
    """Hold an exclusive lock on ``path`` for the ``with`` block.

    ``busy`` is the message of the :class:`SystemExit` raised when another
    process already holds the lock; ``{pid}`` in it is replaced with that
    process's id.

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:  # doctest: +ELLIPSIS
    ...     with exclusive_lock(Path(tmp) / ".x.lock", "busy (pid {pid})"):
    ...         with exclusive_lock(Path(tmp) / ".x.lock", "busy (pid {pid})"):
    ...             pass
    Traceback (most recent call last):
    ...
    SystemExit: busy (pid ...)
    """
    if fcntl is None:
        yield
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if _acquire(fd, path, busy):
            os.ftruncate(fd, 0)
            os.pwrite(fd, f"{os.getpid()}\n".encode("ascii"), 0)
        yield
    finally:
        os.close(fd)
