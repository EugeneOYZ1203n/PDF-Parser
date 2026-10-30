"""Fetch the authors' pretrained DeepTechVec checkpoints (Egiazarian et al.,
hosted on Yandex Disk) and check they load into the vendored network.

    python -m raster_parser.phase_raster_vectorize.deep_tech_vec.download_weights [--prim line|curve|all] [--force]

Writes `raster_parser/weights/deep_tech_vec_lines.weights` (~149 MB) and/or
`deep_tech_vec_curves.weights` (~231 MB). With `--prim line|curve`,
`$DEEPTECHVEC_WEIGHTS_PATH` overrides the destination. An existing file is
kept unless `--force`. Each download is checked against its sha256
(`config.WEIGHTS_SHA256`) before it replaces anything."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from tqdm import tqdm

from raster_parser.commons.logging_setup import configure_logging, get_logger
from raster_parser.phase_raster_vectorize.deep_tech_vec.config import (
    PRIM_KINDS,
    WEIGHTS_ENV_VAR,
    WEIGHTS_FILENAMES,
    WEIGHTS_SHA256,
    WEIGHTS_URLS,
    YANDEX_DOWNLOAD_API,
)

_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"

_LOG = get_logger("deep_tech_vec.download")


_TIMEOUT_S = 60  # per request
_CHUNK_BYTES = 1 << 20  # download read size (1 MiB)


def destination(kind: str, *, use_env: bool) -> Path:
    env = os.environ.get(WEIGHTS_ENV_VAR)
    if use_env and env:
        return Path(env)
    return _WEIGHTS_DIR / WEIGHTS_FILENAMES[kind]


_ATTEMPTS = 6
_USER_AGENT = "raster-parser-download-weights"


def _with_retries(fn, what: str):
    """Yandex Disk answers 503 / times out now and then: retry with backoff."""
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            return fn()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == _ATTEMPTS:
                raise
            wait = 2**attempt
            _LOG.info(f"{what}: {exc} -- retry {attempt}/{_ATTEMPTS - 1} in {wait}s")
            time.sleep(wait)


def _open(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    return urllib.request.urlopen(req, timeout=_TIMEOUT_S)  # noqa: S310 -- fixed https URLs from config / the API


def resolve_href(public_url: str) -> str:
    """Yandex Disk public link -> a direct (one-off) download URL."""
    api = f"{YANDEX_DOWNLOAD_API}?public_key={urllib.parse.quote(public_url, safe='')}"

    def fetch() -> str:
        with _open(api) as resp:
            return json.load(resp)["href"]

    return _with_retries(fetch, "resolving download link")


def download(public_url: str, dest: Path, sha256: str, *, force: bool = False) -> bool:
    """`public_url` -> `dest` via a `.part` file, renamed only once complete
    and its sha256 matches. Returns False when `dest` already existed and was
    kept."""
    if dest.is_file() and not force:
        _LOG.info(f"keeping existing {dest}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    _with_retries(lambda: _fetch(public_url, part, dest.name, sha256), f"downloading {dest.name}")
    shutil.move(str(part), str(dest))
    return True


def _fetch(public_url: str, part: Path, name: str, sha256: str) -> None:
    h = hashlib.sha256()
    with _open(resolve_href(public_url)) as resp, open(part, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=name) as bar:
            while chunk := resp.read(_CHUNK_BYTES):
                f.write(chunk)
                h.update(chunk)
                bar.update(len(chunk))
    if h.hexdigest() != sha256:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"{name}: sha256 {h.hexdigest()} != expected {sha256} (download discarded)"
        )


def verify(path: Path) -> None:
    from raster_parser.phase_raster_vectorize.deep_tech_vec.model.network import load_weights

    model = load_weights(path)
    _LOG.info(
        f"ok: {path.name} -> {model.kind} model, {sum(p.numel() for p in model.parameters()):,} params"
    )


def main(argv: "list[str] | None" = None) -> int:
    configure_logging()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--prim", choices=[*PRIM_KINDS, "all"], default="all", help="which model(s) (default all)"
    )
    ap.add_argument("--force", action="store_true", help="re-download even when the file exists")
    args = ap.parse_args(argv)
    kinds = list(PRIM_KINDS) if args.prim == "all" else [args.prim]
    for kind in kinds:
        dest = destination(kind, use_env=len(kinds) == 1)
        download(WEIGHTS_URLS[kind], dest, WEIGHTS_SHA256[kind], force=args.force)
        verify(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
