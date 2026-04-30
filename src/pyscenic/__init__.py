from importlib.metadata import PackageNotFoundError, version

try:
    if isinstance(__package__, str):
        __version__ = version(__package__)
    else:
        __version__ = "unknown"
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

import logging

from pyscenic.log import create_logging_handler

LOGGER = logging.getLogger(__name__)
# Set logging level.
logging_debug_opt = False
LOGGER.addHandler(create_logging_handler(logging_debug_opt))
LOGGER.setLevel(logging.DEBUG)
