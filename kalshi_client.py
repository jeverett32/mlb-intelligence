"""
Shared Kalshi API authentication helper.
Imported by fetch_kalshi_balance.py, place_bet.py, and any other Kalshi scripts.

Reads credentials from environment variables (loaded from .env by python-dotenv):
    KALSHI_KEY_ID   — your API Key ID from kalshi.com/account/profile
    KALSHI_KEY_PATH — path to your PEM private key file (default: kalshi-key.pem)
    KALSHI_ENV      — "prod" (default) or "demo"
"""

import base64
import os
import time
from pathlib import Path

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv

load_dotenv()

_BASES = {
    "prod": "https://api.elections.kalshi.com/trade-api/v2",
    "demo": "https://demo-api.kalshi.co/trade-api/v2",
}
_PATH_PREFIX = "/trade-api/v2"


def get_base_url(env: str | None = None) -> str:
    env = (env or os.environ.get("KALSHI_ENV", "prod")).lower()
    return _BASES.get(env, _BASES["prod"])


def load_credentials(
    key_id: str | None = None,
    key_path: str | Path | None = None,
    private_key_pem: str | bytes | None = None,
):
    """
    Load the Key ID and RSA private key from environment / .env.
    Returns (key_id: str, private_key: RSAPrivateKey).
    """
    key_id = (key_id or os.environ.get("KALSHI_KEY_ID", "")).strip()
    if not key_id or key_id == "your-key-id-here":
        raise RuntimeError(
            "KALSHI_KEY_ID is not set. Copy .env.example to .env and fill in your Key ID."
        )

    if private_key_pem:
        pem_bytes = (
            private_key_pem
            if isinstance(private_key_pem, bytes)
            else private_key_pem.encode("utf-8")
        )
        private_key = serialization.load_pem_private_key(
            pem_bytes, password=None, backend=default_backend()
        )
        return key_id, private_key

    key_path = Path(key_path or os.environ.get("KALSHI_KEY_PATH", "kalshi-key.pem"))
    if not key_path.exists():
        raise RuntimeError(
            f"Kalshi private key not found at: {key_path}\n"
            "Save your PEM private key (the entire -----BEGIN PRIVATE KEY----- block) "
            f"to {key_path}."
        )

    with open(key_path, "rb") as f:
        private_key = serialization.load_pem_private_key(
            f.read(), password=None, backend=default_backend()
        )

    return key_id, private_key


def auth_headers(key_id, private_key, method: str, path: str) -> dict:
    """
    Build the three Kalshi authentication headers for a request.

    Args:
        key_id:       Your KALSHI-ACCESS-KEY string.
        private_key:  RSA private key object from load_credentials().
        method:       HTTP method in uppercase, e.g. "GET" or "POST".
        path:         Full API path WITHOUT query params, e.g. "/trade-api/v2/portfolio/balance".

    Returns:
        Dict of headers to merge into your requests call.
    """
    ts = str(int(time.time() * 1000))                      # milliseconds
    path_no_query = path.split("?")[0]
    message = f"{ts}{method}{path_no_query}".encode("utf-8")

    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )

    return {
        "KALSHI-ACCESS-KEY":       key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        "Content-Type":            "application/json",
    }


def api_path(endpoint: str) -> str:
    """Return the full path string for signing, e.g. '/trade-api/v2/portfolio/balance'."""
    return f"{_PATH_PREFIX}/{endpoint.lstrip('/')}"
