"""Team-server mode: sign-in, lockout, host allowlist, write guard.

    python3 -m unittest discover -s tests -v
"""

import base64
import http.client
import importlib
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class TeamServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        os.environ["LESSONTALK_USERS"] = os.path.join(cls.tmp, "users.json")
        os.environ["LESSONTALK_HOME"] = os.path.join(cls.tmp, "home")
        os.environ["LESSONTALK_ALLOWED_HOSTS"] = "team.example.ts.net"
        from lessontalk import auth, store, server
        cls.auth = importlib.reload(auth)
        importlib.reload(store)
        cls.server = importlib.reload(server)
        cls.pw = cls.auth.add_user("Asha")
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), cls.server.Handler)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        for k in ("LESSONTALK_USERS", "LESSONTALK_HOME",
                  "LESSONTALK_ALLOWED_HOSTS"):
            os.environ.pop(k, None)

    def req(self, method="GET", path="/api/state", user=None, pw=None,
            host="127.0.0.1", headers=None, body=None, client=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        h = {"Host": host, **(headers or {})}
        if user is not None:
            h["Authorization"] = "Basic " + base64.b64encode(
                f"{user}:{pw}".encode()).decode()
        if client:
            h["X-Forwarded-For"] = client
        c.request(method, path, body=body, headers=h)
        r = c.getresponse()
        return r.status, r.read(), dict(r.getheaders())

    def test_password_is_hashed_and_file_private(self):
        raw = open(os.environ["LESSONTALK_USERS"]).read()
        self.assertNotIn(self.pw, raw)
        self.assertEqual(os.stat(os.environ["LESSONTALK_USERS"]).st_mode & 0o777,
                         0o600)

    def test_no_credentials_gets_401_challenge(self):
        st, _, h = self.req()
        self.assertEqual(st, 401)
        self.assertIn("Basic", h["WWW-Authenticate"])

    def test_right_password_works_and_names_user(self):
        st, body, _ = self.req(user="asha", pw=self.pw)
        self.assertEqual(st, 200)
        s = json.loads(body)
        self.assertTrue(s["hosted"])
        self.assertEqual(s["user"], "asha")
        self.assertIsNone(s["home"])  # no server paths leaked

    def test_wrong_password_and_lockout(self):
        for _ in range(10):
            st, _, _ = self.req(user="asha", pw="nope", client="203.0.113.9")
            self.assertEqual(st, 401)
        st, _, _ = self.req(user="asha", pw=self.pw, client="203.0.113.9")
        self.assertEqual(st, 429)  # even the right password, from that client
        st, _, _ = self.req(user="asha", pw=self.pw, client="203.0.113.10")
        self.assertEqual(st, 200)

    def test_allowed_and_foreign_hosts(self):
        st, _, _ = self.req(user="asha", pw=self.pw, host="team.example.ts.net:8443")
        self.assertEqual(st, 200)
        st, _, _ = self.req(user="asha", pw=self.pw, host="evil.example.com")
        self.assertEqual(st, 403)

    def test_writes_still_need_the_custom_header(self):
        st, _, _ = self.req("POST", "/api/setups", user="asha", pw=self.pw,
                            body=b'{"name":"x"}')
        self.assertEqual(st, 403)

    def test_removed_user_stops_working(self):
        pw = self.auth.add_user("temp")
        self.assertEqual(self.req(user="temp", pw=pw)[0], 200)
        self.auth.remove_user("temp")
        self.assertEqual(self.req(user="temp", pw=pw)[0], 401)

    def test_refuses_public_bind_without_accounts(self):
        saved = self.auth.USERS
        self.auth.USERS = os.path.join(self.tmp, "none.json")
        try:
            with self.assertRaises(SystemExit):
                self.server.serve(0, False, "0.0.0.0")
        finally:
            self.auth.USERS = saved


if __name__ == "__main__":
    unittest.main()
