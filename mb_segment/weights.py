"""Where the weights live, how they are found, downloaded and verified.

Resolution order of the weights directory: ``--weights-dir``, the environment variable ``MB_SEGMENT_WEIGHTS``,
``<package>/../weights`` (a clone of the repository with the release files copied in), ``~/.cache/mb_segment``.
Missing files are downloaded from the GitHub release named in ``weights.json`` and checked against their SHA-256.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY_PATH = os.path.join(HERE, "weights.json")


def registry() -> dict:
    return json.load(open(REGISTRY_PATH))


def model_names() -> List[str]:
    return list(registry()["models"])


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_dir(explicit: Optional[str] = None) -> str:
    for candidate in (explicit, os.environ.get("MB_SEGMENT_WEIGHTS"), os.path.join(HERE, "..", "weights")):
        if candidate and os.path.isdir(candidate):
            return os.path.abspath(candidate)
    cache = os.path.join(os.path.expanduser("~"), ".cache", "mb_segment")
    os.makedirs(cache, exist_ok=True)
    return cache


def _download(url: str, target: str) -> None:
    sys.stderr.write(f"downloading {url}\n")
    tmp = target + ".part"
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                sys.stderr.write(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB")
        sys.stderr.write("\n")
    os.replace(tmp, target)


def ensure_file(name: str, directory: str, expected_sha: Optional[str], download: bool = True) -> str:
    reg = registry()
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        if not download or not reg.get("release_url"):
            raise FileNotFoundError(f"{path} is missing; copy the release files into {directory} or set --weights-dir")
        _download(reg["release_url"].rstrip("/") + "/" + name, path)
    if expected_sha:
        actual = sha256(path)
        if actual != expected_sha:
            raise RuntimeError(f"{path}: SHA-256 {actual} does not match the registry ({expected_sha}); delete the file and retry")
    return path


def resolve(model: str, directory: Optional[str] = None, download: bool = True, with_classifier: bool = True) -> Dict[str, object]:
    reg = registry()
    if model not in reg["models"]:
        raise KeyError(f"unknown model {model!r}; available: {', '.join(reg['models'])}")
    d = weights_dir(directory)
    entry = reg["models"][model]
    members = [ensure_file(f["file"], d, f.get("sha256"), download) for f in entry["members"]]
    classifier = None
    if with_classifier and reg.get("classifier"):
        classifier = ensure_file(reg["classifier"]["file"], d, reg["classifier"].get("sha256"), download)
    return dict(members=members, classifier=classifier, entry=entry, directory=d)
