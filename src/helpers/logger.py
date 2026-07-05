import logging
import os

from dotenv import load_dotenv

load_dotenv()

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    # ponytail: basicConfig configura el root una vez (idempotente); alcanza para app de 1 proceso.
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format=_FORMAT,
        datefmt=_DATEFMT,
    )
    return logging.getLogger(name)
