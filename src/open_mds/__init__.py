__version__ = "0.1.0"

import nltk
from filelock import FileLock
from transformers.utils import is_offline_mode

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            (
                "Please run this package in an online environment first to download nltk data files,"
                " or manually download them with: python -m nltk.downloader punkt"
            )
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)
