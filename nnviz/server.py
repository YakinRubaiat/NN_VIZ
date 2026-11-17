"""Tiny static server for the bundled viewer.

The viewer must be served over HTTP rather than opened as a ``file://`` page:
it fetches ``manifest.json`` and the raw volume binaries, and browsers block
those reads from the filesystem. This keeps that one step painless.

Routes:
    /            -> the viewer (``nnviz/viewer/index.html``)
    /data/...    -> whatever scan directory you point it at
"""

from __future__ import annotations

import functools
import http.server
import os
import socket
import socketserver
import threading
import webbrowser
from pathlib import Path

__all__ = ["serve", "viewer_dir"]


def viewer_dir() -> Path:
    return Path(__file__).resolve().parent / "viewer"


class _Handler(http.server.SimpleHTTPRequestHandler):
    data_root: Path | None = None

    def translate_path(self, path: str) -> str:  # noqa: D102
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/data") and self.data_root is not None:
            rel = clean[len("/data"):].lstrip("/")
            target = (self.data_root / rel).resolve()
            root = self.data_root.resolve()
            if root == target or root in target.parents:
                return str(target)
            return str(root)  # refuse to escape the data root
        return super().translate_path(path)

    def end_headers(self) -> None:  # noqa: D102
        # scans get regenerated constantly; never let the browser cache them
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:  # noqa: D102
        if "404" in (fmt % args):
            print(f"[nnviz] 404 {args[0] if args else ''}")


def _free_port(preferred: int) -> int:
    for candidate in [preferred, *range(preferred + 1, preferred + 25)]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", candidate))
                return candidate
            except OSError:
                continue
    return 0  # let the OS choose


def serve(data_dir=None, *, port: int = 8765, open_browser: bool = True, block: bool = True):
    """Serve the viewer, with ``data_dir`` mounted at ``/data``.

    Returns the ``(httpd, url)`` pair. With ``block=False`` the server runs on a
    daemon thread, which is what you want from a notebook.
    """
    root = viewer_dir()
    if not (root / "index.html").exists():
        raise FileNotFoundError(f"viewer not found at {root}")

    data_root = Path(data_dir).resolve() if data_dir else None
    handler = functools.partial(_Handler, directory=str(root))
    _Handler.data_root = data_root

    chosen = _free_port(port)
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", chosen), handler)
    httpd.daemon_threads = True
    real_port = httpd.server_address[1]
    url = f"http://127.0.0.1:{real_port}/"

    print(f"[nnviz] viewer  {url}")
    if data_root:
        print(f"[nnviz] data    {data_root}  -> /data")
    print("[nnviz] ctrl-c to stop")

    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    if not block:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd, url

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[nnviz] stopped")
    finally:
        httpd.shutdown()
        httpd.server_close()
    return httpd, url
