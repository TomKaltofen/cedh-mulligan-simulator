"""Scryfall API client with local JSON cache.

Rate-limit policy: Scryfall requests 50-100ms between requests.
Cache: one JSON file per card name under .scryfall_cache/.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, cast

CACHE_DIR = ".scryfall_cache"
_API_BASE = "https://api.scryfall.com/cards/named"
_RATE_LIMIT_SECONDS = 0.1

_last_request_time: float = 0.0


def _cache_path(cache_key: str) -> str:
    safe = cache_key.replace("/", "_").replace("\\", "_")
    return os.path.join(CACHE_DIR, f"{safe}.json")


def _read_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    path = _cache_path(cache_key)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return cast(Dict[str, Any], json.load(fh))


def _write_cache(cache_key: str, data: Dict[str, Any]) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_path(cache_key), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def fetch_card(name: str) -> Optional[Dict[str, Any]]:
    """Fetch card data from Scryfall, using the local cache when available.

    Args:
        name: Card display name (any capitalisation, spaces allowed).

    Returns:
        Scryfall card object dict, or ``None`` if the card cannot be found.
    """
    global _last_request_time

    cache_key = name.lower().strip()
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    # Honour rate limit before hitting the API.
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _RATE_LIMIT_SECONDS:
        time.sleep(_RATE_LIMIT_SECONDS - elapsed)

    headers = {"User-Agent": "cedh-mulligan-simulator/1.0 (open-source MTG tool)"}

    for param_type in ("exact", "fuzzy"):
        params = urllib.parse.urlencode({param_type: name})
        url = f"{_API_BASE}?{params}"
        req = urllib.request.Request(url, headers=headers)  # nosec B310
        try:
            with urllib.request.urlopen(req) as response:  # nosec B310
                raw: Dict[str, Any] = cast(Dict[str, Any], json.loads(response.read().decode("utf-8")))
                _last_request_time = time.monotonic()
                _write_cache(cache_key, raw)
                return raw
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                if param_type == "fuzzy":
                    return None  # Neither exact nor fuzzy found anything.
                continue  # Try fuzzy fallback.
            return None
        except urllib.error.URLError:
            return None

    return None
