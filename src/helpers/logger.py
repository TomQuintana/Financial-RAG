import logging
import os

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_SUPPRESS_DEFAULT = "openai,httpcore,httpx,fpdf,azure,urllib3.connectionpool"


class _LevelFilter(logging.Filter):
    def __init__(self, levels: set[int]) -> None:
        self._levels = levels

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno in self._levels


def get_logger(name: str) -> logging.Logger:
    # ponytail: basicConfig no soporta filter en handler; setup manual igual de simple.
    raw = os.getenv("LOG_LEVEL", "INFO")

    levels = {
        getattr(logging, lvl.strip().upper(), logging.INFO) for lvl in raw.split(",")
    }

    root = logging.getLogger()

    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
        handler.addFilter(_LevelFilter(levels))
        root.addHandler(handler)
        root.setLevel(min(levels))

        suppress = os.getenv("LOG_LEVEL_SUPPRESS", _SUPPRESS_DEFAULT)
        for library in suppress.split(","):
            if lib := library.strip():
                logging.getLogger(lib).setLevel(logging.WARNING)

    return logging.getLogger(name)
