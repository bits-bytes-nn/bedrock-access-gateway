"""Tests for the checks that keep caller-supplied image urls from reaching internal endpoints."""

import base64
import http.server
import socket
import threading
import time

import pytest
from fastapi import HTTPException

from api import image_url as image_url_module
from api.image_url import normalize_image_content_type, parse_image_url

PUBLIC_IP = "93.184.216.34"


class FakeResponse:
    """Minimal stand-in for urllib3.BaseHTTPResponse as the fetch path uses it."""

    def __init__(self, status=200, headers=None, body=b"", location=None):
        self.status = status
        self.headers = headers if headers is not None else {}
        self._body = body
        self.released = False
        if location:
            self.headers["Location"] = location

    def stream(self, chunk_size):
        for start in range(0, max(len(self._body), 1), chunk_size):
            yield self._body[start : start + chunk_size]

    def release_conn(self):
        self.released = True


@pytest.fixture
def resolve_to_public(monkeypatch):
    """Resolve every hostname to a public address so only the url logic is exercised."""

    def fake_getaddrinfo(host, port, **_kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (PUBLIC_IP, port))]

    monkeypatch.setattr(image_url_module.socket, "getaddrinfo", fake_getaddrinfo)


@pytest.fixture
def capture_request(monkeypatch):
    """Record the targets the gateway connects to and return canned responses."""
    calls = []
    responses = []

    def fake_open(target, read_timeout):
        calls.append(target)
        return responses.pop(0) if responses else FakeResponse(body=b"image-bytes")

    monkeypatch.setattr(image_url_module, "_open", fake_open)
    return calls, responses


@pytest.fixture
def forbid_request(monkeypatch):
    """Fail the test if an outbound request is attempted."""

    def fail_open(*_args, **_kwargs):
        raise AssertionError("no outbound request expected")

    monkeypatch.setattr(image_url_module, "_open", fail_open)


@pytest.mark.parametrize(
    "url",
    [
        # ECS task credential endpoint and EC2 instance metadata service.
        "http://169.254.170.2/v2/credentials/abc",
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
        "http://[fd00:ec2::254]/latest/meta-data/",
        # Loopback.
        "http://127.0.0.1:8080/health",
        "http://127.1/health",
        "http://[::1]:8080/health",
        # Private and otherwise non-routable ranges.
        "http://10.0.0.5/",
        "http://172.16.3.4/",
        "http://192.168.1.1/",
        "http://100.64.0.1/",
        "http://[fd12:3456::1]/",
        "http://0.0.0.0/",
        # IPv4-mapped IPv6 form of the metadata endpoint.
        "http://[::ffff:169.254.169.254]/latest/meta-data/",
    ],
)
def test_rejects_non_routable_addresses(url):
    with pytest.raises(HTTPException) as exc_info:
        parse_image_url(url)
    assert exc_info.value.status_code == 400


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "file://localhost/etc/shadow",
        "gopher://127.0.0.1:6379/_INFO",
        "ftp://192.168.0.1/image.png",
        "dict://127.0.0.1:11211/stats",
        "http:/169.254.169.254/",
        "not-a-url",
        "",
    ],
)
def test_rejects_unsupported_schemes(url):
    with pytest.raises(HTTPException) as exc_info:
        parse_image_url(url)
    assert exc_info.value.status_code == 400


def test_blocked_url_is_rejected_before_any_request(forbid_request):
    """Validation must happen before the request, not after it has been sent."""
    with pytest.raises(HTTPException):
        parse_image_url("http://169.254.169.254/latest/meta-data/iam/security-credentials/")


def test_error_for_blocked_address_does_not_reveal_the_target():
    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("http://169.254.170.2/v2/credentials/abc")
    assert "169.254" not in exc_info.value.detail


def test_unresolvable_host_is_rejected(monkeypatch):
    def fake_getaddrinfo(*_args, **_kwargs):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(image_url_module.socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("https://example.com/cat.png")
    assert exc_info.value.status_code == 400


def test_fetches_a_public_url(resolve_to_public, capture_request):
    calls, responses = capture_request
    responses.append(FakeResponse(headers={"Content-Type": "image/png"}, body=b"png-bytes"))

    data, content_type = parse_image_url("https://example.com/cat.png")

    assert (data, content_type) == (b"png-bytes", "image/png")
    target = calls[0]
    # The connection goes to the address that was checked, under the original host name.
    assert (target.scheme, target.host, target.ips, target.port) == ("https", "example.com", (PUBLIC_IP,), 443)
    assert (target.request_uri, target.host_header) == ("/cat.png", "example.com")


def test_redirect_to_a_public_url_is_followed(resolve_to_public, capture_request):
    calls, responses = capture_request
    responses.append(FakeResponse(status=302, location="https://cdn.example.com/cat.png"))
    responses.append(FakeResponse(headers={"Content-Type": "image/webp"}, body=b"webp-bytes"))

    data, content_type = parse_image_url("https://example.com/cat.png")

    assert (data, content_type) == (b"webp-bytes", "image/webp")
    assert [call.host for call in calls] == ["example.com", "cdn.example.com"]


def test_redirect_to_the_metadata_endpoint_is_blocked(monkeypatch, capture_request):
    """A public host must not be able to redirect the gateway to an internal address."""
    _calls, responses = capture_request
    responses.append(FakeResponse(status=302, location="http://169.254.169.254/latest/meta-data/"))

    def fake_getaddrinfo(host, port, **_kwargs):
        if host == "example.com":
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (PUBLIC_IP, port))]
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (host, port))]

    monkeypatch.setattr(image_url_module.socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("https://example.com/cat.png")
    assert exc_info.value.status_code == 400


def test_redirect_loop_is_bounded(resolve_to_public, monkeypatch):
    attempts = []

    def fake_open(target, read_timeout):
        attempts.append(target)
        return FakeResponse(status=302, location="https://example.com/next")

    monkeypatch.setattr(image_url_module, "_open", fake_open)

    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("https://example.com/cat.png")
    assert exc_info.value.status_code == 400
    assert len(attempts) == image_url_module.MAX_REDIRECTS + 1


def test_host_allowlist_rejects_other_hosts(resolve_to_public, capture_request, monkeypatch):
    monkeypatch.setattr(image_url_module, "IMAGE_URL_ALLOWED_HOSTS", frozenset({"images.example.com"}))

    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("https://evil.example.net/cat.png")
    assert exc_info.value.status_code == 400

    _data, content_type = parse_image_url("https://images.example.com/cat.png")
    assert content_type == "image/jpeg"


def test_fetching_can_be_disabled(monkeypatch):
    monkeypatch.setattr(image_url_module, "ENABLE_IMAGE_URL_FETCH", False)
    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("https://example.com/cat.png")
    assert exc_info.value.status_code == 400


def test_oversized_body_is_rejected(resolve_to_public, capture_request, monkeypatch):
    monkeypatch.setattr(image_url_module, "IMAGE_URL_MAX_SIZE_MB", 1)
    _calls, responses = capture_request
    responses.append(FakeResponse(headers={"Content-Type": "image/png"}, body=b"x" * (2 * 1024 * 1024)))

    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("https://example.com/big.png")
    assert exc_info.value.status_code == 413


def test_oversized_content_length_is_rejected_before_reading(resolve_to_public, capture_request, monkeypatch):
    monkeypatch.setattr(image_url_module, "IMAGE_URL_MAX_SIZE_MB", 1)
    _calls, responses = capture_request
    responses.append(
        FakeResponse(
            headers={"Content-Type": "image/png", "Content-Length": str(50 * 1024 * 1024)},
            body=b"",
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("https://example.com/big.png")
    assert exc_info.value.status_code == 413


def test_non_200_response_is_rejected(resolve_to_public, capture_request):
    _calls, responses = capture_request
    responses.append(FakeResponse(status=404))

    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("https://example.com/missing.png")
    assert exc_info.value.status_code == 400


def test_connection_error_is_reported_as_a_client_error(resolve_to_public, monkeypatch):
    def fake_open(*_args, **_kwargs):
        raise image_url_module.urllib3.exceptions.NewConnectionError(None, "refused")

    monkeypatch.setattr(image_url_module, "_open", fake_open)

    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("https://example.com/cat.png")
    assert exc_info.value.status_code == 400


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("image/png", "image/png"),
        ("image/PNG", "image/png"),
        ("image/png; charset=binary", "image/png"),
        ("image/jpg", "image/jpeg"),
        ("image/gif", "image/gif"),
        ("image/webp", "image/webp"),
        # Anything Bedrock cannot take falls back to the default format.
        ("image/svg+xml", "image/jpeg"),
        ("text/html", "image/jpeg"),
        (None, "image/jpeg"),
        ("", "image/jpeg"),
    ],
)
def test_content_type_is_normalized_to_a_supported_format(raw, expected):
    assert normalize_image_content_type(raw) == expected


def test_missing_content_type_header_falls_back(resolve_to_public, capture_request):
    """A response with no Content-Type must not raise; the previous code crashed here."""
    _calls, responses = capture_request
    responses.append(FakeResponse(body=b"bytes"))

    data, content_type = parse_image_url("https://example.com/cat.png")
    assert (data, content_type) == (b"bytes", "image/jpeg")


def test_data_url_is_decoded_without_fetching(forbid_request):
    payload = base64.b64encode(b"png-bytes").decode()
    assert parse_image_url(f"data:image/png;base64,{payload}") == (b"png-bytes", "image/png")


def test_data_url_payload_may_be_wrapped(forbid_request):
    payload = base64.b64encode(b"png-bytes" * 20).decode()
    wrapped = "\n".join(payload[i : i + 16] for i in range(0, len(payload), 16))
    data, content_type = parse_image_url(f"data:image/png;base64,\n{wrapped}")
    assert (data, content_type) == (b"png-bytes" * 20, "image/png")


def test_invalid_data_url_payload_is_a_client_error(forbid_request):
    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("data:image/png;base64,not!valid!base64")
    assert exc_info.value.status_code == 400


@pytest.mark.parametrize(
    "url",
    [
        # urlparse sees example.com here, while urllib3 connects to the address before the backslash.
        "http://169.254.169.254\\@example.com/../latest/meta-data/",
        "http://10.0.0.5:8080\\@example.com/../admin",
        "http://evil.test\\@images.example.com/x",
        "http://example.com\\.169.254.169.254/",
        "http://exa mple.com/",
        "http://example.com/\ncat.png",
    ],
)
def test_urls_that_parsers_disagree_on_are_rejected(url, resolve_to_public, forbid_request):
    with pytest.raises(HTTPException) as exc_info:
        parse_image_url(url)
    assert exc_info.value.status_code == 400


def test_credentials_in_url_are_rejected(resolve_to_public, forbid_request):
    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("http://169.254.169.254@example.com/cat.png")
    assert exc_info.value.status_code == 400


def test_allowlist_is_checked_against_the_host_actually_contacted(resolve_to_public, capture_request, monkeypatch):
    monkeypatch.setattr(image_url_module, "IMAGE_URL_ALLOWED_HOSTS", frozenset({"images.example.com"}))
    calls, _responses = capture_request

    parse_image_url("https://IMAGES.example.com:8443/cat.png")

    assert (calls[0].host, calls[0].port, calls[0].host_header) == (
        "images.example.com",
        8443,
        "images.example.com:8443",
    )


def test_connection_is_pinned_to_the_checked_address(monkeypatch, capture_request):
    """DNS rebinding: a second lookup must not be able to swap in an internal address."""
    answers = iter([PUBLIC_IP, "169.254.169.254"])

    def rebinding_getaddrinfo(host, port, **_kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (next(answers), port))]

    monkeypatch.setattr(image_url_module.socket, "getaddrinfo", rebinding_getaddrinfo)
    calls, _responses = capture_request

    parse_image_url("https://rebind.example.com/cat.png")

    assert [call.ips for call in calls] == [(PUBLIC_IP,)]


@pytest.mark.parametrize(
    "address",
    [
        "64:ff9b::a9fe:a9fe",  # NAT64 form of 169.254.169.254
        "64:ff9b::a00:5",  # NAT64 form of 10.0.0.5
        "2002:a9fe:a9fe::1",  # 6to4 form of 169.254.169.254
        "::a9fe:a9fe",  # IPv4-compatible form of 169.254.169.254
    ],
)
def test_ipv6_transition_forms_of_internal_addresses_are_blocked(address, monkeypatch, forbid_request):
    def fake_getaddrinfo(host, port, **_kwargs):
        return [(socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port, 0, 0))]

    monkeypatch.setattr(image_url_module.socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("https://v6.example.com/cat.png")
    assert exc_info.value.status_code == 400


def test_slow_download_is_cut_off_at_the_deadline(resolve_to_public, capture_request, monkeypatch):
    _calls, responses = capture_request
    responses.append(FakeResponse(headers={"Content-Type": "image/png"}, body=b"x" * (3 * 64 * 1024)))
    # Start, before connecting, then past the deadline from the first chunk on.
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(
        image_url_module, "_monotonic", lambda: next(ticks, image_url_module.TOTAL_TIMEOUT_SECONDS + 1.0)
    )

    with pytest.raises(HTTPException) as exc_info:
        parse_image_url("https://example.com/slow.png")
    assert exc_info.value.status_code == 400


def test_response_connection_is_released(resolve_to_public, capture_request):
    _calls, responses = capture_request
    response = FakeResponse(status=404)
    responses.append(response)

    with pytest.raises(HTTPException):
        parse_image_url("https://example.com/missing.png")
    assert response.released


# --- real sockets on loopback, exercising _open and the deadline -------------------


class _Handler(http.server.BaseHTTPRequestHandler):
    seen_hosts: list = []
    drip_delay = 0.0

    def do_GET(self):
        _Handler.seen_hosts.append(self.headers.get("Host"))
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.end_headers()
        if _Handler.drip_delay:
            # About 5 seconds in total, well past the 1 second deadline the test sets.
            for _ in range(50):
                try:
                    self.wfile.write(b"x")
                    self.wfile.flush()
                except OSError:
                    return  # the client shut the connection down
                time.sleep(_Handler.drip_delay)
        else:
            self.wfile.write(b"png-bytes")

    def log_message(self, *_args):
        pass


@pytest.fixture
def loopback_server():
    _Handler.seen_hosts = []
    _Handler.drip_delay = 0.0
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _target(port, ips=("127.0.0.1",)):
    return image_url_module._Target(
        scheme="http", host="images.example.com", port=port, ips=ips, request_uri="/cat.png"
    )


def test_open_connects_to_the_pinned_address_with_the_original_host(loopback_server):
    port = loopback_server.server_address[1]
    response = image_url_module._open(_target(port), read_timeout=5)
    assert response.read() == b"png-bytes"
    response.release_conn()
    assert _Handler.seen_hosts == [f"images.example.com:{port}"]


def test_open_falls_back_to_the_next_address(loopback_server, monkeypatch):
    port = loopback_server.server_address[1]
    attempted = []
    real_new_conn = image_url_module._HTTPConnection._new_conn

    def refuse_first_address(conn):
        attempted.append(conn.host)
        if conn.host == "192.0.2.1":
            raise image_url_module.NewConnectionError(conn, "refused")
        return real_new_conn(conn)

    monkeypatch.setattr(image_url_module._HTTPConnection, "_new_conn", refuse_first_address)
    response = image_url_module._open(_target(port, ips=("192.0.2.1", "127.0.0.1")), read_timeout=2)
    assert attempted == ["192.0.2.1", "127.0.0.1"]
    assert response.read() == b"png-bytes"
    response.release_conn()


def test_trickling_server_is_cut_off_by_the_deadline(loopback_server, monkeypatch):
    """One byte at a time keeps every recv under the read timeout; the deadline must still apply."""
    _Handler.drip_delay = 0.1
    port = loopback_server.server_address[1]
    monkeypatch.setattr(image_url_module, "TOTAL_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(image_url_module, "_validate_url", lambda url: _target(port))

    started = time.monotonic()
    with pytest.raises(HTTPException) as exc_info:
        parse_image_url(f"http://images.example.com:{port}/cat.png")
    assert exc_info.value.status_code == 400
    # Without the deadline the download would take the full 5 seconds.
    assert time.monotonic() - started < 4
