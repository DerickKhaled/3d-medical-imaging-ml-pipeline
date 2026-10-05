"""SHA-256 hashes for files, arrays and JSON.

Everything in the pipeline is identified by the hash of its content, not by its
name or date. That is what lets the trace check that nothing was changed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

_CHUNK_BYTES = 1 << 20


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    """Hash of a JSON-serialisable object, independent of key order."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_bytes(canonical.encode("utf-8"))


def sha256_array(array: np.ndarray) -> str:
    """Hash of array *content*: dtype and shape are part of the identity.

    Used for predictions instead of a file hash, because compressed NIfTI files
    embed gzip timestamps and the same mask would get a different file hash on
    every save.
    """
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def short_hash(full_hash: str, length: int = 8) -> str:
    return full_hash[:length]
