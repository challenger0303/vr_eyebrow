"""
OSCQuery: discover VRChat (_oscjson._tcp + /?HOST_INFO) and publish a local OSCQuery service.

Spec: https://github.com/Vidvox/OSCQueryProposal
"""
from __future__ import annotations

import json
import random
import socket
import string
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from zeroconf import IPVersion, ServiceBrowser, ServiceInfo, ServiceListener, Zeroconf
except ImportError:  # pragma: no cover
    Zeroconf = None  # type: ignore
    ServiceBrowser = None  # type: ignore
    ServiceInfo = None  # type: ignore
    ServiceListener = object  # type: ignore
    IPVersion = None  # type: ignore

OSCQUERY_SERVICE_TYPE = "_oscjson._tcp.local."
DEFAULT_VRCHAT_OSC_PORT = 9000


def zeroconf_available() -> bool:
    return Zeroconf is not None and ServiceBrowser is not None


def _http_get_json(host: str, port: int, path_query: str, timeout: float = 2.0) -> Optional[Any]:
    q = path_query.lstrip("/")
    url = f"http://{host}:{port}/{q}" if q else f"http://{host}:{port}/"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        if not raw:
            return None
        return json.loads(raw.decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError, ValueError):
        return None


def _ipv4_from_service_info(info: "ServiceInfo") -> str:
    if not info.addresses:
        return "127.0.0.1"
    return socket.inet_ntoa(info.addresses[0])


def discover_vrchat_osc_target(wait_seconds: float = 3.0) -> Optional[Tuple[str, int]]:
    """
    Browse _oscjson._tcp, fetch /?HOST_INFO, prefer VRChat-named services.
    Returns (osc_ip, osc_port) for UDP sends into VRChat, or None.
    """
    if not zeroconf_available():
        return None

    candidates: List[Tuple[str, int, str]] = []
    lock = threading.Lock()

    class _L(ServiceListener):
        def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            info = zc.get_service_info(type_, name, timeout=3000)
            if info is None:
                return
            host = _ipv4_from_service_info(info)
            with lock:
                candidates.append((host, int(info.port), name))

        def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            pass

        def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            pass

    zc = Zeroconf(ip_version=IPVersion.V4Only)
    listener = _L()
    browser = ServiceBrowser(zc, OSCQUERY_SERVICE_TYPE, listener)
    try:
        time.sleep(max(0.5, wait_seconds))
    finally:
        browser.cancel()
        zc.close()

    with lock:
        snapshot = list(candidates)

    def score(name: str, hi: Optional[dict]) -> int:
        s = 0
        n = name.lower()
        if "vrchat" in n:
            s += 10
        if hi:
            nm = str(hi.get("NAME", "")).lower()
            if "vrchat" in nm:
                s += 20
        return s

    best: Optional[Tuple[int, str, int, Optional[dict]]] = None

    for host, http_port, mdns_name in snapshot:
        hi = _http_get_json(host, http_port, "/?HOST_INFO", timeout=2.0)
        hi_d = hi if isinstance(hi, dict) else None
        sc = score(mdns_name, hi_d)
        if best is None or sc > best[0]:
            best = (sc, host, http_port, hi_d)

    if best is None:
        return None

    _, _http_host, _http_port, hi = best
    osc_ip = "127.0.0.1"
    osc_port = DEFAULT_VRCHAT_OSC_PORT
    if hi:
        oi = hi.get("OSC_IP")
        if isinstance(oi, str) and oi.strip():
            osc_ip = oi.strip()
        op = hi.get("OSC_PORT")
        if isinstance(op, int) and 1 <= op <= 65535:
            osc_port = op
        elif isinstance(op, float) and op == int(op):
            p = int(op)
            if 1 <= p <= 65535:
                osc_port = p

    if osc_ip in ("0.0.0.0", "::", "[::]"):
        osc_ip = "127.0.0.1"

    return osc_ip, osc_port


def _walk_to_path(root_doc: Dict[str, Any], want_path: str) -> Optional[Dict[str, Any]]:
    want = "/" + want_path.strip("/")
    parts = [p for p in want.strip("/").split("/") if p]
    node: Any = root_doc
    if node.get("FULL_PATH") != "/":
        return None
    for seg in parts:
        cont = node.get("CONTENTS")
        if not isinstance(cont, dict) or seg not in cont:
            return None
        node = cont[seg]
        if not isinstance(node, dict):
            return None
    if isinstance(node, dict) and node.get("FULL_PATH") == want:
        return node
    return None


class OscQueryPublisher:
    """
    Local OSCQuery HTTP server + _oscjson._tcp advertisement.
    get_values: map short parameter name -> float (e.g. BrowExpressionLeft).
    """

    def __init__(
        self,
        display_name: str,
        full_paths: List[str],
        get_values: Callable[[], Dict[str, float]],
    ):
        self._display_name = display_name
        self._full_paths = sorted(full_paths)
        self._get_values = get_values
        self._udp_sock: Optional[socket.socket] = None
        self._http: Optional[ThreadingHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None
        self._zc: Any = None
        self._svc_info: Any = None

    def _host_info(self) -> Dict[str, Any]:
        osc_port = DEFAULT_VRCHAT_OSC_PORT
        if self._udp_sock:
            osc_port = int(self._udp_sock.getsockname()[1])
        return {
            "NAME": self._display_name,
            "OSC_PORT": osc_port,
            "OSC_TRANSPORT": "UDP",
            "EXTENSIONS": {"ACCESS": True, "VALUE": True, "RANGE": True},
        }

    def _leaf_node(self, full_path: str, val: float) -> Dict[str, Any]:
        node: Dict[str, Any] = {
            "FULL_PATH": full_path,
            "TYPE": "f",
            "ACCESS": 3,
            "VALUE": [float(val)],
        }
        short = full_path.rsplit("/", 1)[-1]
        if short in ("BrowExpressionLeft", "BrowExpressionRight"):
            node["RANGE"] = [{"MIN": -1.0, "MAX": 1.0}]
        else:
            node["RANGE"] = [{"MIN": 0.0, "MAX": 1.0}]
        return node

    def _build_contents_tree(self, vals: Dict[str, float]) -> Dict[str, Any]:
        tree: Dict[str, Any] = {}

        def insert(parts: List[str], full: str, short_key: str) -> None:
            d = tree
            acc: List[str] = []
            for i, seg in enumerate(parts):
                acc.append(seg)
                acc_path = "/" + "/".join(acc)
                last = i == len(parts) - 1
                if last:
                    d[seg] = self._leaf_node(full, float(vals.get(short_key, 0.0)))
                else:
                    if seg not in d or "CONTENTS" not in d[seg]:
                        d[seg] = {"FULL_PATH": acc_path, "CONTENTS": {}}
                    d = d[seg]["CONTENTS"]

        for fp in self._full_paths:
            parts = [p for p in fp.strip("/").split("/") if p]
            if not parts:
                continue
            short = parts[-1]
            insert(parts, fp, short)
        return tree

    def _root_document(self) -> Dict[str, Any]:
        vals = dict(self._get_values())
        return {"FULL_PATH": "/", "CONTENTS": self._build_contents_tree(vals)}

    def start(self) -> None:
        if not zeroconf_available():
            raise RuntimeError("zeroconf is not installed")

        self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_sock.bind(("127.0.0.1", 0))

        pub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                query = parsed.query
                path = urllib.parse.unquote(parsed.path or "/")
                if path != "/" and not path.startswith("/"):
                    path = "/" + path

                if query == "HOST_INFO":
                    body = json.dumps(pub._host_info()).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if query and query != "HOST_INFO":
                    self.send_error(400, "Unsupported query")
                    return

                if path in ("/", ""):
                    doc = pub._root_document()
                    body = json.dumps(doc).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if path in pub._full_paths:
                    short = path.rsplit("/", 1)[-1]
                    vals = pub._get_values()
                    doc = pub._leaf_node(path, float(vals.get(short, 0.0)))
                    body = json.dumps(doc).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                full = pub._root_document()
                node = _walk_to_path(full, path)
                if node is not None:
                    body = json.dumps(node).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                self.send_error(404)

        class _S(ThreadingHTTPServer):
            pass

        self._http = _S(("127.0.0.1", 0), Handler)
        self._http_thread = threading.Thread(target=self._http.serve_forever, daemon=True)
        self._http_thread.start()
        tcp_port = int(self._http.server_address[1])

        suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
        svc_name = f"{self._display_name}-{suffix}._oscjson._tcp.local."
        self._zc = Zeroconf(ip_version=IPVersion.V4Only)
        self._svc_info = ServiceInfo(
            OSCQUERY_SERVICE_TYPE,
            svc_name,
            addresses=[socket.inet_aton("127.0.0.1")],
            port=tcp_port,
            properties={},
            server="vr-eyebrow.local.",
        )
        self._zc.register_service(self._svc_info)

    def stop(self) -> None:
        if self._zc and self._svc_info:
            try:
                self._zc.unregister_service(self._svc_info)
            except OSError:
                pass
            try:
                self._zc.close()
            except OSError:
                pass
        self._zc = None
        self._svc_info = None

        if self._http:
            try:
                self._http.shutdown()
            except Exception:
                pass
            try:
                self._http.server_close()
            except Exception:
                pass
        self._http = None
        self._http_thread = None

        if self._udp_sock:
            try:
                self._udp_sock.close()
            except OSError:
                pass
        self._udp_sock = None
