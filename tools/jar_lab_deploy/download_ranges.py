"""Resumable, hash-checked ModelScope API range downloads on one chosen NIC."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import socket
import struct
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection
import urllib3.util.connection


def download_file(repo, record, directory, token, interface=None, workers=12):
    # The target's IPv6 ModelScope route stalls; this affects only this process.
    urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET
    root = Path(directory).resolve()
    target = (root / record["path"]).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Download escapes bundle directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    size = record["size"]
    def checksum(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if target.is_file() and target.stat().st_size == size and checksum(target) == record["sha256"]:
        return target
    class Adapter(HTTPAdapter):
        def __init__(self, bound_interface):
            self.bound_interface = bound_interface
            super().__init__()

        def init_poolmanager(self, *args, **kwargs):
            options = [option for option in HTTPConnection.default_socket_options
                       if option[:2] != (socket.SOL_SOCKET, socket.SO_BINDTODEVICE)]
            if self.bound_interface:
                options.append((socket.SOL_SOCKET, socket.SO_BINDTODEVICE, self.bound_interface.encode() + b"\0"))
            kwargs["socket_options"] = options
            return super().init_poolmanager(*args, **kwargs)
    def available_interface(attempt):
        # DHCP renewals can temporarily remove an address. Retry over the default
        # route on alternate attempts, without changing any host network setting.
        if not interface or attempt % 2:
            return None
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                fcntl.ioctl(probe.fileno(), 0x8915, struct.pack("256s", interface.encode()[:15]))
            return interface
        except OSError:
            return None
    part_size = 64 * 1024 * 1024
    ranges = [(start, min(size - 1, start + part_size - 1)) for start in range(0, size, part_size)]
    def fetch(bounds):
        start, end = bounds
        part = Path(str(target) + f"_{start}_{end}")
        expected = end - start + 1
        with requests.Session() as session:
            session.trust_env = False
            for attempt in range(12):
                have = part.stat().st_size if part.exists() else 0
                if have == expected:
                    return part
                if have > expected:
                    raise ValueError("Range part exceeds expected size")
                try:
                    session.get_adapter("https://").close()
                    session.mount("https://", Adapter(available_interface(attempt)))
                    first = start + have
                    with session.get(f"https://modelscope.cn/api/v1/models/{repo}/repo",
                                     params={"Revision": "master", "FilePath": record["path"]},
                                     headers={"Authorization": "Bearer " + token, "Range": f"bytes={first}-{end}"},
                                     stream=True, timeout=(30, 180)) as response:
                        response.raise_for_status()
                        exact_range = response.status_code == 206 and response.headers.get("Content-Range") == f"bytes {first}-{end}/{size}"
                        complete_object = response.status_code == 200 and first == 0 and end == size - 1
                        if not (exact_range or complete_object):
                            raise ValueError("ModelScope did not honor exact byte range")
                        with part.open("ab") as output:
                            for block in response.iter_content(1024 * 1024):
                                if output.tell() + len(block) > expected:
                                    raise ValueError("Range response exceeds expected size")
                                output.write(block)
                    if part.stat().st_size != expected:
                        raise OSError("Incomplete byte range")
                    print(json.dumps(dict(event="range_complete", path=record["path"], start=start, bytes=expected)), flush=True)
                    return part
                except (requests.RequestException, OSError) as error:
                    print(json.dumps(dict(event="range_retry", path=record["path"], start=start,
                                          attempt=attempt + 1, error=type(error).__name__)), flush=True)
                    time.sleep(min(10, attempt + 1))
            raise RuntimeError("ModelScope range retries exhausted")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(fetch, ranges))
    temporary = target.with_name(target.name + ".incomplete")
    with temporary.open("wb") as output:
        for part in parts:
            with part.open("rb") as stream:
                shutil.copyfileobj(stream, output, 8 * 1024 * 1024)
    if temporary.stat().st_size != size or checksum(temporary) != record["sha256"]:
        raise ValueError("ModelScope download checksum mismatch")
    temporary.replace(target)
    # Parts belong to this downloader (or its SDK predecessor in the same new bundle).
    for part in parts:
        part.unlink()
    print(json.dumps(dict(event="file_verified", path=record["path"], bytes=size, sha256=record["sha256"])), flush=True)
    return target
