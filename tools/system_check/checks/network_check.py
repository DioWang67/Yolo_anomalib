"""Network checks for the optional inspection-sync outbox.

Inspection always commits to local SQLite first; server sync is an opt-in
background outbox that ships disabled. So the default outcome here is ``SKIP``,
and a station with no network is not a failure.

When sync *is* enabled, the checks are deliberately shallow: resolve the host,
open a TCP connection, close it. No HTTP request is sent, because a request
carrying the station's credentials would be an outward-facing action, and
because an empty POST to a production inspection endpoint is not something a
diagnostic tool should do uninvited.

The API token is never read, logged, or included in any report — only whether
the named environment variable is set.
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

from tools.system_check.context import AppContext
from tools.system_check.results import CheckResult, Status

_CONNECT_TIMEOUT_S = 5.0
_DEFAULT_PORTS = {"https": 443, "http": 80}


def check(context: AppContext) -> list[CheckResult]:
    """Report sync reachability, or skip when sync is disabled."""
    if not context.sync_enabled():
        return [
            CheckResult(
                check_id="network.sync",
                title="Inspection sync",
                status=Status.SKIP,
                detail=(
                    "inspection_sync_enabled is false. Inspections commit to "
                    "local SQLite and no network access is required."
                ),
                requirement="Outbound HTTPS only when sync is enabled",
                measured="disabled",
                source="config.example.yaml:80-87",
            )
        ]

    endpoint = str(context.config.get("inspection_sync_endpoint") or "")
    results = [_endpoint_result(context, endpoint)]
    results.append(_token_result(context))
    return results


def _endpoint_result(context: AppContext, endpoint: str) -> CheckResult:
    """Resolve and TCP-connect to the configured sync endpoint."""
    if not endpoint:
        return CheckResult(
            check_id="network.sync",
            title="Inspection sync",
            status=Status.FAIL,
            detail="Sync is enabled but inspection_sync_endpoint is empty.",
            requirement="A reachable HTTPS endpoint",
            measured="not configured",
            source="config.example.yaml:81",
            remedy="Set inspection_sync_endpoint, or disable inspection_sync_enabled.",
        )

    parsed = urlparse(endpoint)
    host = parsed.hostname
    scheme = (parsed.scheme or "").lower()
    if not host:
        return CheckResult(
            check_id="network.sync",
            title="Inspection sync",
            status=Status.FAIL,
            detail=f"inspection_sync_endpoint is not a usable URL: {endpoint}",
            requirement="A reachable HTTPS endpoint",
            measured="unparseable",
            source="config.example.yaml:81",
            remedy="Correct the endpoint URL.",
        )

    allow_http = bool(context.config.get("inspection_sync_allow_insecure_http", False))
    if scheme == "http" and not allow_http:
        return CheckResult(
            check_id="network.sync",
            title="Inspection sync",
            status=Status.FAIL,
            detail=(
                "The endpoint uses http:// while "
                "inspection_sync_allow_insecure_http is false, so the outbox "
                "will refuse to send."
            ),
            requirement="https:// endpoint",
            measured="http://",
            source="config.example.yaml:87",
            remedy="Use an https:// endpoint.",
        )

    port = parsed.port or _DEFAULT_PORTS.get(scheme, 443)
    timeout = _configured_timeout(context)

    try:
        addresses = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        return CheckResult(
            check_id="network.sync",
            title="Inspection sync",
            status=Status.FAIL,
            detail=f"DNS resolution for {host} failed: {exc}",
            requirement=f"{host}:{port} reachable",
            measured="DNS failure",
            source="config.example.yaml:81",
            remedy="Check DNS and the station's network configuration.",
            data={"host": host, "port": port},
        )

    family, socktype, proto, _, sockaddr = addresses[0]
    try:
        with socket.socket(family, socktype, proto) as connection:
            connection.settimeout(timeout)
            connection.connect(sockaddr)
    except OSError as exc:
        return CheckResult(
            check_id="network.sync",
            title="Inspection sync",
            status=Status.WARNING,
            detail=(
                f"{host}:{port} resolved but the TCP connection failed: {exc}. "
                "Inspections still commit locally; the outbox retries after an "
                "outage, so this degrades traceability rather than inspection."
            ),
            requirement=f"{host}:{port} reachable",
            measured="connect failed",
            source="config.example.yaml:80-87",
            remedy="Ask IT to open outbound access from this station.",
            data={"host": host, "port": port},
        )

    return CheckResult(
        check_id="network.sync",
        title="Inspection sync",
        status=Status.PASS,
        detail=(
            f"{host}:{port} accepted a TCP connection within {timeout:g}s. No "
            "HTTP request was sent."
        ),
        requirement=f"{host}:{port} reachable",
        measured="reachable",
        source="config.example.yaml:80-87",
        data={"host": host, "port": port},
    )


def _configured_timeout(context: AppContext) -> float:
    """Return the station's sync timeout, bounded for a preflight probe."""
    raw = context.config.get("inspection_sync_timeout_seconds", _CONNECT_TIMEOUT_S)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _CONNECT_TIMEOUT_S
    return min(max(value, 1.0), 30.0)


def _token_result(context: AppContext) -> CheckResult:
    """Confirm the credential variable is set, without reading its value."""
    variable = str(
        context.config.get("inspection_sync_api_token_env") or "YOLO11_INSPECTION_SYNC_TOKEN"
    )
    present = bool(os.environ.get(variable))
    return CheckResult(
        check_id="network.sync_token",
        title="Inspection sync credential",
        status=Status.PASS if present else Status.FAIL,
        detail=(
            f"Environment variable {variable} is set. Its value is never read "
            "or reported by this tool."
            if present
            else (
                f"Sync is enabled but {variable} is not set in this "
                "environment, so the outbox cannot authenticate."
            )
        ),
        requirement=f"{variable} set in the service environment",
        measured="set" if present else "not set",
        source="config.example.yaml:82",
        remedy=(
            None
            if present
            else f"Set {variable} for the account that runs the application."
        ),
        data={"variable": variable, "present": present},
    )
