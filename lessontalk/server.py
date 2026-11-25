"""Local web app. Binds to 127.0.0.1 only: the recording never leaves the
computer, and nothing on the network can reach the app.

    python3 -m lessontalk serve
"""

from __future__ import annotations

import json
import os
import re
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from . import auth, store
from .audio import AudioError

LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
# Extra hostnames the app may be reached by, e.g. a Tailscale Funnel name.
ALLOWED_HOSTS = LOCAL_HOSTS | {
    h.strip().lower() for h in
    os.environ.get("LESSONTALK_ALLOWED_HOSTS", "").split(",") if h.strip()}

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.html")
MAX_UPLOAD = 4 << 30  # 4 GiB, a 2-hour 48 kHz 24-bit stereo recording is ~2 GiB
TYPES = {".html": "text/html; charset=utf-8", ".json": "application/json",
         ".wav": "audio/wav"}


class Handler(BaseHTTPRequestHandler):
    server_version = "LessonTalk"

    def log_message(self, fmt, *args):  # quiet; this is a desktop app
        pass

    # -------------------------------------------------------------- guards
    def _host_ok(self) -> bool:
        """Refuse requests whose Host is not this machine, which blocks DNS
        rebinding from a web page in the same browser."""
        host = (self.headers.get("Host") or "").lower()
        if not host.startswith("["):
            host = host.rsplit(":", 1)[0]
        else:
            host = host.split("]")[0] + "]"
        return host in ALLOWED_HOSTS

    def _client(self) -> str:
        return (self.headers.get("X-Forwarded-For") or
                self.client_address[0]).split(",")[0].strip()

    def _authed(self) -> bool:
        """Every request needs an account once any account exists. Sends the
        401 or 429 itself and returns False when the request must stop."""
        if not auth.enabled():
            self.user = None
            return True
        self.user, locked = auth.check(self.headers.get("Authorization"),
                                       self._client())
        if self.user:
            return True
        if locked:
            self._err(429, "too many failed sign-ins; try again in 10 minutes")
        else:
            self._send(401, b"Sign in with the account you were given.",
                       "text/plain; charset=utf-8",
                       {"WWW-Authenticate": 'Basic realm="Lesson Talk", '
                                            'charset="UTF-8"'})
        return False

    def _write_ok(self) -> bool:
        """Mutating calls need a custom header. A cross-site form or fetch
        cannot send one without a CORS preflight, which this server never
        approves, so other web pages cannot drive the app."""
        return self.headers.get("X-LessonTalk") == "1"

    # ------------------------------------------------------------- replies
    def _send(self, code: int, body: bytes, ctype: str,
              extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _err(self, code: int, msg: str) -> None:
        self._json({"error": msg}, code)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 1 << 20:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def _file(self, path: str) -> None:
        """Serve a file, honouring Range so the audio player can seek."""
        if not os.path.isfile(path):
            return self._err(404, "not found")
        size = os.path.getsize(path)
        ctype = TYPES.get(os.path.splitext(path)[1], "application/octet-stream")
        rng = self.headers.get("Range")
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng or "")
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start, end = max(0, size - int(m.group(2))), size - 1
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            code, length = 206, end - start + 1
        else:
            start, length, code = 0, size, 200
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if code == 206:
            self.send_header("Content-Range",
                             f"bytes {start}-{start + length - 1}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as fh:
            fh.seek(start)
            left = length
            try:
                while left > 0:
                    buf = fh.read(min(1 << 16, left))
                    if not buf:
                        break
                    self.wfile.write(buf)
                    left -= len(buf)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _state(self) -> dict:
        hosted = auth.enabled()
        return {"setups": store.setups(), "lessons": store.all_lessons(),
                "home": None if hosted else store.HOME,
                "hosted": hosted, "user": getattr(self, "user", None)}

    # -------------------------------------------------------------- routes
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self._authed():
            return
        if not self._host_ok():
            return self._err(403, "forbidden")
        path = urlparse(self.path).path
        if path == "/":
            with open(APP, "rb") as fh:
                return self._send(200, fh.read(), TYPES[".html"])
        if path == "/api/state":
            return self._json(self._state())
        m = re.fullmatch(r"/lessons/([0-9a-z-]+)/"
                         r"(card\.html|result\.json|(?:stereo|mix|near|far)\.wav)",
                         path)
        if m:
            try:
                d = store.lesson_dir(m.group(1))
            except KeyError:
                return self._err(404, "not found")
            return self._file(os.path.join(d, m.group(2)))
        self._err(404, "not found")

    def do_PUT(self):
        if not self._authed():
            return
        if not (self._host_ok() and self._write_ok()):
            return self._err(403, "forbidden")
        m = re.fullmatch(r"/api/lessons/([0-9a-z-]+)/audio/(stereo|near|far)",
                         urlparse(self.path).path)
        if not m:
            return self._err(404, "not found")
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n <= MAX_UPLOAD:
            return self._err(413, "file is empty or larger than 4 GB")
        try:
            store.save_upload(m.group(1), m.group(2), self.rfile, n)
        except KeyError:
            return self._err(404, "no such lesson")
        except ValueError as e:
            return self._err(400, str(e))
        self._json({"ok": True})

    def do_POST(self):
        if not self._authed():
            return
        if not (self._host_ok() and self._write_ok()):
            return self._err(403, "forbidden")
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/lessons":
                return self._json(store.create(
                    body.get("label", ""), body.get("group", ""),
                    body.get("setup", "default"),
                    body.get("recorded_on", ""), body.get("mode", "stereo"),
                    int(body.get("teacher_channel", 0))))
            if path == "/api/setups":
                return self._json({"id": store.add_setup(body.get("name", ""))})
            m = re.fullmatch(r"/api/lessons/([0-9a-z-]+)/"
                             r"(analyse|swap|calibrate|delete-audio)", path)
            if not m:
                return self._err(404, "not found")
            lid, act = m.groups()
            if act == "analyse":
                store.analyse_async(lid)
            elif act == "swap":
                store.swap(lid)
            elif act == "calibrate":
                store.calibrate(lid, float(body["teacher_share_pct"]))
            elif act == "delete-audio":
                store.delete_audio(lid)
            return self._json(self._state())
        except KeyError:
            return self._err(404, "no such lesson")
        except (ValueError, AudioError, json.JSONDecodeError) as e:
            return self._err(400, str(e))

    def do_DELETE(self):
        if not self._authed():
            return
        if not (self._host_ok() and self._write_ok()):
            return self._err(403, "forbidden")
        m = re.fullmatch(r"/api/lessons/([0-9a-z-]+)", urlparse(self.path).path)
        if not m:
            return self._err(404, "not found")
        try:
            store.delete(m.group(1))
        except KeyError:
            return self._err(404, "no such lesson")
        self._json(self._state())


def serve(port: int = 8765, open_browser: bool = True,
          host: str = "127.0.0.1") -> None:
    if host not in ("127.0.0.1", "localhost", "::1") and not auth.enabled():
        raise SystemExit(
            f"refusing to listen on {host} with no accounts: anyone who can "
            "reach it could read and delete lessons. Create one first with "
            "`lesson-talk add-user NAME`.")
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Lesson Talk is running at {url}")
    print(f"Lessons are kept in {store.HOME}")
    if auth.enabled():
        print(f"Sign-in required: {len(auth.users())} account(s)")
    print("Press Ctrl+C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
