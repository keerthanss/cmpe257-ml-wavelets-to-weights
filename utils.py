"""
utils.py — logging setup and small shared helpers.
"""

import logging
import sys
import time
import functools
from config import LOG_FILE

_LOG_FMT = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger that writes to both stdout and LOG_FILE.
    Calling this multiple times with the same name is safe — handlers
    are not duplicated.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_LOG_FMT)
    logger.addHandler(ch)

    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_LOG_FMT)
    logger.addHandler(fh)

    return logger


def init_run_logging(log_file: str):
    """Redirect all existing loggers' file handlers to *log_file*.

    Called once at the start of a run so every module's logger writes
    to the per-run log file instead of the default.
    """
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_LOG_FMT)

    for logger in [logging.getLogger(n)
                   for n in logging.Logger.manager.loggerDict]:
        for h in logger.handlers[:]:
            if isinstance(h, logging.FileHandler) and not isinstance(h, logging.StreamHandler):
                logger.removeHandler(h)
        logger.addHandler(fh)


def timed(logger: logging.Logger):
    """Decorator — logs entry, exit, and wall-clock time of a function."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            logger.info(f"→ {fn.__name__} started")
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            logger.info(f"✓ {fn.__name__} finished in {elapsed:.1f}s")
            return result
        return wrapper
    return decorator


def section(logger: logging.Logger, title: str):
    """Log a visual section separator."""
    bar = "─" * 60
    logger.info(bar)
    logger.info(f"  {title}")
    logger.info(bar)


def fmt_elapsed(seconds: float) -> str:
    """Format seconds into ``Xm YY.Zs`` or ``YY.Zs``."""
    m, s = divmod(seconds, 60)
    if m >= 1:
        return f"{int(m)}m {s:04.1f}s"
    return f"{s:.1f}s"
