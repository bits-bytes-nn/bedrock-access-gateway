"""Safe retrieval of caller-supplied image URLs.

The OpenAI chat completions schema lets an ``image_url`` content part carry
either an inline ``data:`` URI or a remote URL. Fetching a remote URL means the
gateway performs an outbound request on behalf of the caller, so an unvalidated
URL turns the gateway into a request proxy: a caller could point it at endpoints
that are only reachable from the task itself -- the ECS container credential
endpoint (``169.254.170.2``), the EC2 instance metadata service
(``169.254.169.254``), ``localhost``, or other VPC-internal services -- and the
fetched bytes would be sent to the model and echoed back in the completion.

To prevent that (server-side request forgery), a remote URL is only fetched when
all of the following hold:

* the scheme is ``http`` or ``https``;
* the host is in ``IMAGE_URL_ALLOWED_HOSTS`` when that allowlist is configured;
* every address the host resolves to is globally routable, which excludes the
  link-local range used by the metadata endpoints as well as private, loopback,
  shared-address-space, reserved, and multicast ranges;
* the response body stays within ``IMAGE_URL_MAX_SIZE_MB``.

Redirects are followed manually so that every hop is re-validated instead of
letting a public URL redirect to an internal one.

The URL is parsed once, with the same parser that builds the outgoing request,
and the connection is opened to the exact address that passed the checks (with
the original host name kept for the ``Host`` header and TLS verification). This
rules out two bypasses: a URL that two parsers read differently (for example
``http://169.254.169.254\\@example.com/``), and DNS rebinding, where a host
resolves to a public address for the check and an internal one for the
connection. Each fetch also has an overall deadline so a slow server cannot hold
a worker indefinitely.
"""

import base64
import binascii
import ipaddress
import logging
import re
import socket
import threading
import time
from dataclasses import dataclass
from urllib.parse import urljoin

import certifi
import urllib3
from fastapi import HTTPException
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.exceptions import ConnectTimeoutError, HTTPError, LocationParseError, NewConnectionError
from urllib3.util import parse_url

from api.setting import (
    ENABLE_IMAGE_URL_FETCH,
    IMAGE_URL_ALLOWED_HOSTS,
    IMAGE_URL_MAX_SIZE_MB,
)

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = frozenset({"http", "https"})
DEFAULT_PORTS = {"http": 80, "https": 443}

# Image formats accepted by the Bedrock Converse API.
# Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ImageBlock.html
SUPPORTED_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
DEFAULT_IMAGE_TYPE = "image/jpeg"
# Clock used for fetch deadlines; replaced in tests.
_monotonic = time.monotonic

# Content types that name a supported format under a different label.
IMAGE_TYPE_ALIASES = {"image/jpg": "image/jpeg"}

CONNECT_TIMEOUT_SECONDS = 5
READ_TIMEOUT_SECONDS = 30
# Upper bound for a whole fetch, redirects included.
TOTAL_TIMEOUT_SECONDS = 60
MAX_REDIRECTS = 3
CHUNK_SIZE = 64 * 1024
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

# NAT64 (RFC 6052) embeds an IPv4 address in the last 32 bits.
_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")
# Deprecated IPv4-compatible IPv6 addresses (::a.b.c.d).
_IPV4_COMPATIBLE_PREFIX = ipaddress.ip_network("::/96")

DATA_URL_PATTERN = r"^data:(image/[a-z]*);base64,\s*"

# Returned for every failure that depends on where the URL points, so that error
# messages cannot be used to probe which internal hosts exist.
_UNREACHABLE_DETAIL = "Unable to access the image url"


def normalize_image_content_type(content_type: str | None) -> str:
    """Map a content type onto an image format Bedrock accepts."""
    # Drop any parameters, e.g. "image/png; charset=binary".
    media_type = (content_type or "").split(";")[0].strip().lower()
    media_type = IMAGE_TYPE_ALIASES.get(media_type, media_type)
    if media_type in SUPPORTED_IMAGE_TYPES:
        return media_type
    return DEFAULT_IMAGE_TYPE


@dataclass(frozen=True)
class _Target:
    """A validated fetch target, pinned to the address that passed the checks."""

    scheme: str
    host: str
    port: int
    # Every address the host resolved to, all of which passed the checks.
    ips: tuple[str, ...]
    request_uri: str

    @property
    def host_header(self) -> str:
        return self.host if self.port == DEFAULT_PORTS[self.scheme] else f"{self.host}:{self.port}"


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """Return the IPv4 address an IPv6 transition address forwards to, if any."""
    if ip.ipv4_mapped:
        return ip.ipv4_mapped
    if ip.sixtofour:
        return ip.sixtofour
    if ip in _NAT64_PREFIX or ip in _IPV4_COMPATIBLE_PREFIX:
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return None


def _resolved_addresses(host: str, port: int) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Resolve a host to every address it maps to, IP literals included."""
    try:
        addr_infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError):
        logger.warning("Unable to resolve image url host")
        raise HTTPException(status_code=400, detail=_UNREACHABLE_DETAIL) from None

    return [ipaddress.ip_address(addr_info[4][0]) for addr_info in addr_infos]


def _is_fetchable(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether an address is publicly routable, looking through IPv6 transition forms."""
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = _embedded_ipv4(ip)
        if embedded is not None:
            # ::ffff:169.254.169.254, 2002:a9fe:a9fe::, 64:ff9b::a9fe:a9fe etc.
            # reach the same host as the IPv4 address they wrap.
            return embedded.is_global
    return ip.is_global


def _validate_url(url: str) -> _Target:
    """Check an image URL and return the pinned target the gateway may fetch."""
    # Backslashes, whitespace and control characters are read differently by
    # different URL parsers, so they are never needed in a legitimate image URL.
    if "\\" in url or any(ch.isspace() or ord(ch) < 0x20 or ord(ch) == 0x7F for ch in url):
        raise HTTPException(status_code=400, detail="Image url contains invalid characters")

    try:
        parsed = parse_url(url)
    except LocationParseError:
        raise HTTPException(status_code=400, detail="Image url is not valid") from None

    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise HTTPException(
            status_code=400,
            detail="Image url must be an http(s) url or a base64 data url",
        )
    if parsed.auth:
        raise HTTPException(status_code=400, detail="Image url must not contain credentials")

    host = (parsed.host or "").lower()
    if not host:
        raise HTTPException(status_code=400, detail="Image url is missing a host")

    if IMAGE_URL_ALLOWED_HOSTS and host.strip("[]") not in IMAGE_URL_ALLOWED_HOSTS:
        raise HTTPException(
            status_code=400,
            detail=f"Image url host is not allowed: {host}",
        )

    port = parsed.port or DEFAULT_PORTS[scheme]
    addresses = _resolved_addresses(host.strip("[]"), port)
    for ip in addresses:
        if not _is_fetchable(ip):
            logger.warning("Blocked image url resolving to non-routable address %s", ip)
            raise HTTPException(status_code=400, detail=_UNREACHABLE_DETAIL)

    return _Target(
        scheme=scheme,
        host=host,
        port=port,
        ips=tuple(dict.fromkeys(str(ip) for ip in addresses)),
        request_uri=parsed.request_uri,
    )


class _Deadline:
    """Shut down the sockets of a fetch once its overall time budget is spent.

    Socket timeouts only bound a single read, so a server that sends a byte just
    before each timeout could otherwise hold a worker thread indefinitely.
    """

    def __init__(self, seconds: float):
        self._sockets: list[socket.socket] = []
        self._lock = threading.Lock()
        self.expired = False
        self._timer = threading.Timer(seconds, self._expire)
        self._timer.daemon = True

    def __enter__(self) -> "_Deadline":
        _current.deadline = self
        self._timer.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._timer.cancel()
        _current.deadline = None

    def register(self, sock: socket.socket) -> None:
        with self._lock:
            self._sockets.append(sock)
            if self.expired:
                self._shutdown(sock)

    def _expire(self) -> None:
        with self._lock:
            self.expired = True
            for sock in self._sockets:
                self._shutdown(sock)

    @staticmethod
    def _shutdown(sock: socket.socket) -> None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


# The deadline of the fetch running on this thread; connections register with it.
_current = threading.local()


class _DeadlineConnectionMixin:
    def _new_conn(self) -> socket.socket:
        sock = super()._new_conn()
        deadline = getattr(_current, "deadline", None)
        if deadline is not None:
            deadline.register(sock)
        return sock


class _HTTPConnection(_DeadlineConnectionMixin, HTTPConnection):
    pass


class _HTTPSConnection(_DeadlineConnectionMixin, HTTPSConnection):
    pass


class _HTTPConnectionPool(urllib3.HTTPConnectionPool):
    ConnectionCls = _HTTPConnection


class _HTTPSConnectionPool(urllib3.HTTPSConnectionPool):
    ConnectionCls = _HTTPSConnection


def _open(target: _Target, read_timeout: float) -> urllib3.BaseHTTPResponse:
    """Send the request to a validated address without re-resolving the host.

    Addresses are tried in resolution order, like a normal client would, so an
    unreachable IPv6 address does not prevent falling back to IPv4.
    """
    timeout = urllib3.Timeout(connect=min(CONNECT_TIMEOUT_SECONDS, read_timeout), read=read_timeout)
    hostname = target.host.strip("[]")
    for attempt, ip in enumerate(target.ips, start=1):
        if target.scheme == "https":
            pool = _HTTPSConnectionPool(
                ip,
                target.port,
                timeout=timeout,
                retries=False,
                cert_reqs="CERT_REQUIRED",
                ca_certs=certifi.where(),
                # Verify the certificate and send SNI for the requested host, not the IP.
                server_hostname=hostname,
                assert_hostname=hostname,
            )
        else:
            pool = _HTTPConnectionPool(ip, target.port, timeout=timeout, retries=False)
        try:
            return pool.urlopen(
                "GET",
                target.request_uri,
                headers={"Host": target.host_header, "Accept": "image/*"},
                redirect=False,
                preload_content=False,
            )
        except (NewConnectionError, ConnectTimeoutError):
            if attempt == len(target.ips):
                raise
    raise HTTPException(status_code=400, detail=_UNREACHABLE_DETAIL)


def _read_capped(response: urllib3.BaseHTTPResponse, max_bytes: int, deadline: float) -> bytes:
    """Read a response body, refusing anything larger than ``max_bytes`` or slower than the deadline."""
    declared_length = response.headers.get("Content-Length")
    if declared_length and declared_length.isdigit() and int(declared_length) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the maximum size of {IMAGE_URL_MAX_SIZE_MB} MB",
        )

    content = bytearray()
    for chunk in response.stream(CHUNK_SIZE):
        content += chunk
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds the maximum size of {IMAGE_URL_MAX_SIZE_MB} MB",
            )
        if _monotonic() > deadline:
            break
    # Checked after the loop as well: when the deadline shuts the socket down, a body
    # without Content-Length simply ends early and must not be returned as complete.
    if _monotonic() > deadline:
        logger.warning("Image url download exceeded %s seconds", TOTAL_TIMEOUT_SECONDS)
        raise HTTPException(status_code=400, detail=_UNREACHABLE_DETAIL)
    return bytes(content)


def parse_image_url(image_url: str) -> tuple[bytes, str]:
    """Resolve an image url to a tuple of (image data, content type).

    Handles both an inline base64 ``data:`` url and, subject to the checks in
    this module, a remote http(s) url.
    """
    match = re.search(DATA_URL_PATTERN, image_url)
    if match:
        # Clients sometimes wrap the payload across lines, so drop whitespace
        # before decoding rather than letting it fail validation.
        payload = re.sub(r"\s+", "", re.sub(DATA_URL_PATTERN, "", image_url))
        try:
            image_data = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=400, detail="Image data url is not valid base64") from None
        return image_data, normalize_image_content_type(match.group(1))

    return _fetch_image(image_url)


def _fetch_image(image_url: str) -> tuple[bytes, str]:
    """Fetch a remote image url, returning a tuple of (image data, content type)."""
    if not ENABLE_IMAGE_URL_FETCH:
        raise HTTPException(
            status_code=400,
            detail="Fetching images by url is disabled, please pass a base64 data url instead",
        )

    max_bytes = IMAGE_URL_MAX_SIZE_MB * 1024 * 1024
    deadline = _monotonic() + TOTAL_TIMEOUT_SECONDS
    with _Deadline(TOTAL_TIMEOUT_SECONDS):
        url = image_url
        for _ in range(MAX_REDIRECTS + 1):
            target = _validate_url(url)
            remaining = deadline - _monotonic()
            if remaining <= 0:
                raise HTTPException(status_code=400, detail=_UNREACHABLE_DETAIL)
            try:
                response = _open(target, read_timeout=min(READ_TIMEOUT_SECONDS, remaining))
            except (HTTPError, OSError):
                logger.warning("Image url request failed", exc_info=True)
                raise HTTPException(status_code=400, detail=_UNREACHABLE_DETAIL) from None

            try:
                if response.status in REDIRECT_STATUSES:
                    location = response.headers.get("Location")
                    if not location:
                        raise HTTPException(status_code=400, detail=_UNREACHABLE_DETAIL)
                    # Followed here, not by the client, so that every hop is re-validated.
                    url = urljoin(url, location)
                    continue

                if response.status != 200:
                    raise HTTPException(status_code=400, detail=_UNREACHABLE_DETAIL)

                content_type = normalize_image_content_type(response.headers.get("Content-Type"))
                try:
                    return _read_capped(response, max_bytes, deadline), content_type
                except (HTTPError, OSError):
                    logger.warning("Image url download failed", exc_info=True)
                    raise HTTPException(status_code=400, detail=_UNREACHABLE_DETAIL) from None
            finally:
                response.release_conn()

        raise HTTPException(status_code=400, detail="Image url has too many redirects")
