"""Manifest-driven dataset download with SHA-256 verification."""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import urllib.error
import urllib.request
from pathlib import Path


class UnpublishedRelease(RuntimeError):
    pass


def read_manifest(path: str | Path) -> dict:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported data manifest schema")
    return data


def sha256(path: str | Path, chunk_size=8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def selected_archives(manifest: dict, profile: str, scan: int | None = None):
    profiles = manifest.get("profiles", {})
    if profile not in profiles:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(profiles)}")
    names = list(profiles[profile])
    if scan is not None:
        suffix = f"-scan{int(scan)}"
        names = [name for name in names if name.endswith(suffix)]
        if not names:
            raise ValueError(f"profile {profile!r} has no scan {scan}")
    archives = manifest.get("archives", {})
    return [(name, archives[name]) for name in names]


def _download(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as error:
        if error.code == 416 and offset:
            partial.replace(destination)
            return
        raise
    append = offset and response.status == 206
    mode = "ab" if append else "wb"
    with response, partial.open(mode) as output:
        shutil.copyfileobj(response, output, length=8 << 20)
    partial.replace(destination)


def _safe_destination(root: Path, name: str) -> Path:
    destination = (root / name).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"unsafe archive path: {name}") from error
    return destination


def _extract(archive_path: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    if archive_path.name.endswith(".tar.zst"):
        import zstandard
        with archive_path.open("rb") as compressed:
            with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as archive:
                    for member in archive:
                        _safe_destination(destination, member.name)
                        if member.issym() or member.islnk() or member.isdev():
                            raise ValueError(f"unsupported archive member: {member.name}")
                        archive.extract(member, destination)
    else:
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive.getmembers():
                _safe_destination(destination, member.name)
                if member.issym() or member.islnk() or member.isdev():
                    raise ValueError(f"unsupported archive member: {member.name}")
            archive.extractall(destination)


def fetch(manifest_path, destination, profile="quickstart", scan=None, extract=False):
    manifest = read_manifest(manifest_path)
    if not manifest.get("published", False):
        browser = manifest.get("browser_url") or "not assigned"
        raise UnpublishedRelease(
            "The data release is private or has not been published yet. "
            f"Release page: {browser}"
        )
    destination = Path(destination).expanduser().resolve()
    results = []
    for name, item in selected_archives(manifest, profile, scan):
        url = item.get("direct_url")
        if not url:
            raise UnpublishedRelease(
                f"{name} has not been published yet and has no direct URL. "
                f"Current release status: {manifest.get('browser_url') or 'not assigned'}"
            )
        output = destination / item["filename"]
        if not output.exists() or sha256(output) != item["sha256"]:
            _download(url, output)
        actual = sha256(output)
        if actual != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {output}: {actual}")
        results.append(output)
        if extract:
            _extract(output, destination / item.get("extract_to", name))
    return results


def verify(manifest_path, directory, profile="quickstart", scan=None):
    manifest = read_manifest(manifest_path)
    directory = Path(directory)
    results = []
    for name, item in selected_archives(manifest, profile, scan):
        path = directory / item["filename"]
        if not path.exists():
            results.append((name, False, "missing"))
            continue
        actual = sha256(path)
        results.append((name, actual == item.get("sha256"), actual))
    return results
