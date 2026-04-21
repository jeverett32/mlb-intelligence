"""Discord webhook alerter.

Reads MLB_ALERT_WEBHOOK from the environment. If not set, becomes a no-op so
local dev and tests never spam. Swallows all exceptions — alerting must never
take down the caller.
"""
import json
import logging
import os
import urllib.request

_LOG = logging.getLogger(__name__)

_MAX_MSG_LEN = 1900  # Discord hard limit is 2000


def send(content: str) -> None:
    url = os.environ.get("MLB_ALERT_WEBHOOK")
    if not url:
        return
    if len(content) > _MAX_MSG_LEN:
        content = content[:_MAX_MSG_LEN] + "…"
    payload = json.dumps({"content": content}).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()
    except Exception as e:
        _LOG.warning("alert webhook failed: %s", e)
