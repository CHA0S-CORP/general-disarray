"""
Identity Verification
=====================
Optional caller-identity verification with two factors:

- a **static PIN** (something the caller knows), and
- a **rolling TOTP code** (RFC 6238, from the caller's authenticator app).

Credentials resolve **per-caller first, then a global fallback**: a caller
enrolled in data/verify_credentials.json uses their own PIN/secret; otherwise
the agent falls back to the global ``VERIFY_PIN`` / ``VERIFY_TOTP_SECRET`` from
config. Empty global factors and no enrolled caller means the feature is simply
off — verification can't be attempted and nothing changes.

Two objects:
- ``VerificationStore`` — persistence of per-caller credentials, modelled on
  ``PersonaStore`` (single JSON object, one lock, atomic write, fail-open). PINs
  are stored only as a salted PBKDF2-SHA256 hash; TOTP secrets are stored as the
  base32 shared secret (needed to recompute codes).
- ``IdentityVerifier`` — the pure check logic (constant-time PIN compare, TOTP
  verify with a skew window), used by both the VERIFY tool and the REST API.

Fail-open throughout (a broken store never breaks a call); the caller is simply
treated as unverified. Callers that gate on the result must fail *closed*.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import pyotp
except Exception:  # pragma: no cover - dependency guard (mirrors house style)
    pyotp = None

# Caller ids become keys in a shared JSON file — constrain them exactly like
# caller_memory does (no path separators, bounded length).
_SAFE_CALLER_ID = re.compile(r"[A-Za-z0-9._+-]{1,64}")

# PBKDF2 work factor. Generous for a short numeric PIN; verification is off the
# hot path (once per call, or per REST request).
_PBKDF2_ITERATIONS = 200_000
_PBKDF2_DIGEST = "sha256"

# Supported TOTP hash algorithms (RFC 6238). Keyed by canonical upper-case name.
_TOTP_ALGORITHMS = {
    "SHA1": hashlib.sha1,
    "SHA256": hashlib.sha256,
    "SHA512": hashlib.sha512,
}
_TOTP_DEFAULTS = {"digits": 6, "period": 30, "algorithm": "SHA1"}


def resolve_totp_digest(algorithm: Optional[str]):
    """Map an algorithm name (SHA1/SHA256/SHA512, dashes/case-insensitive) to its
    hashlib constructor, defaulting to SHA1 for anything unrecognised."""
    key = (algorithm or "SHA1").upper().replace("-", "")
    return _TOTP_ALGORITHMS.get(key, hashlib.sha1)


def totp_params(config, digits: Optional[int] = None, period: Optional[int] = None,
                algorithm: Optional[str] = None) -> Dict[str, Any]:
    """Resolve the TOTP construction parameters: explicit override, else the
    configured default, else the RFC baseline (6 digits / 30s / SHA1)."""
    return {
        "digits": int(digits or getattr(config, "verify_totp_digits", 0)
                      or _TOTP_DEFAULTS["digits"]),
        "period": int(period or getattr(config, "verify_totp_period", 0)
                      or _TOTP_DEFAULTS["period"]),
        "algorithm": str(algorithm or getattr(config, "verify_totp_algorithm", "")
                         or _TOTP_DEFAULTS["algorithm"]),
    }


def build_totp(secret: str, params: Dict[str, Any]):
    """Construct a ``pyotp.TOTP`` from a resolved params dict (see totp_params)."""
    return pyotp.TOTP(secret, digits=int(params["digits"]),
                      digest=resolve_totp_digest(params["algorithm"]),
                      interval=int(params["period"]))


def is_safe_caller_id(caller_id: str) -> bool:
    return bool(caller_id) and _SAFE_CALLER_ID.fullmatch(caller_id) is not None


def _hash_pin(pin: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac(
        _PBKDF2_DIGEST, pin.encode("utf-8"), salt, _PBKDF2_ITERATIONS
    ).hex()


def _const_eq(a: str, b: str) -> bool:
    """Constant-time string equality that tolerates non-ASCII input.

    ``hmac.compare_digest`` raises TypeError for ``str`` operands containing
    non-ASCII characters (e.g. a full-width digit pasted into a PIN field);
    comparing the UTF-8 bytes keeps a bad candidate a plain mismatch.
    """
    return hmac.compare_digest((a or "").encode("utf-8"), (b or "").encode("utf-8"))


class VerificationStore:
    """Per-caller PIN/TOTP credentials in data/verify_credentials.json.

    Single JSON object ``{caller_id: {pin_hash, pin_salt, totp_secret, ...}}``.
    Thread-safe and fail-open, mirroring PersonaStore. The raw PIN is never
    stored — only its salted PBKDF2 hash.
    """

    def __init__(self, config):
        self.config = config
        self.path: Path = getattr(config, "verify_credentials_file", None) or (
            Path(getattr(config, "data_dir", Path("./data"))) / "verify_credentials.json")
        self._lock = threading.Lock()

    # -- disk ------------------------------------------------------------
    def _load_raw(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if isinstance(v, dict)}
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Could not read verify store {self.path}: {e}")
        return {}

    def _write_raw(self, data: Dict[str, Dict[str, Any]]) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            # Plaintext TOTP secrets live here: owner-only from the first byte
            # (0600 at creation, so there is no world-readable window).
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            try:
                os.chmod(tmp, 0o600)  # in case the tmp file pre-existed with wider bits
            except OSError:
                pass
            tmp.replace(self.path)
            return True
        except Exception as e:
            logger.warning(f"Could not write verify store {self.path}: {e}")
            return False

    # -- API -------------------------------------------------------------
    def get(self, caller_id: str) -> Optional[Dict[str, Any]]:
        if not is_safe_caller_id(caller_id):
            return None
        with self._lock:
            return self._load_raw().get(caller_id)

    def set_credentials(self, caller_id: str, pin: Optional[str] = None,
                        totp_secret: Optional[str] = None,
                        generate_totp: bool = False) -> Optional[Dict[str, Any]]:
        """Enroll/update a caller's PIN and/or TOTP secret. Returns the public
        view of the stored record, or None on invalid input / write failure.

        A non-empty ``pin`` sets/rotates the PIN. ``generate_totp`` mints a fresh
        base32 secret; otherwise a non-empty ``totp_secret`` is stored verbatim.
        Existing factors are preserved when their argument is omitted.
        """
        if not is_safe_caller_id(caller_id):
            return None
        with self._lock:
            data = self._load_raw()
            record = dict(data.get(caller_id) or {})

            if pin:
                salt = secrets.token_bytes(16)
                record["pin_salt"] = salt.hex()
                record["pin_hash"] = _hash_pin(pin, salt)

            if generate_totp:
                if pyotp is None:
                    logger.warning("generate_totp requested but pyotp is unavailable")
                    return None
                record["totp_secret"] = pyotp.random_base32()
            elif totp_secret:
                cleaned = totp_secret.strip().replace(" ", "").upper()
                if not re.fullmatch(r"[A-Z2-7]+=*", cleaned):
                    logger.warning("Rejected non-base32 TOTP secret for %s", caller_id)
                    return None
                record["totp_secret"] = cleaned

            if not record.get("pin_hash") and not record.get("totp_secret"):
                # Nothing to store — don't create an empty enrollment.
                return None

            record["updated_at"] = datetime.now(timezone.utc).isoformat()
            data[caller_id] = record
            if not self._write_raw(data):
                return None
            return self.public_view(caller_id, record)

    def delete(self, caller_id: str) -> bool:
        if not is_safe_caller_id(caller_id):
            return False
        with self._lock:
            data = self._load_raw()
            if caller_id not in data:
                return False
            del data[caller_id]
            return self._write_raw(data)

    def public_view(self, caller_id: str,
                    record: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Metadata safe to return over the API — never the secret or PIN hash."""
        if record is None:
            record = self.get(caller_id) or {}
        return {
            "caller_id": caller_id,
            "has_pin": bool(record.get("pin_hash")),
            "has_totp": bool(record.get("totp_secret")),
            "updated_at": record.get("updated_at"),
        }

    def provisioning_uri(self, caller_id: str, issuer: str) -> Optional[str]:
        """otpauth:// URI for the caller's per-caller secret (for QR/enrollment),
        or None when there's no per-caller secret or pyotp is missing."""
        if pyotp is None:
            return None
        record = self.get(caller_id)
        secret = (record or {}).get("totp_secret")
        if not secret:
            return None
        try:
            totp = build_totp(secret, totp_params(self.config))
            return totp.provisioning_uri(name=caller_id, issuer_name=issuer)
        except Exception as e:
            logger.warning(f"provisioning_uri failed for {caller_id}: {e}")
            return None


class IdentityVerifier:
    """Pure verification logic over the store + global config fallback.

    Per-caller credentials take precedence: if a caller has enrolled a PIN, only
    that PIN is accepted (no silent fall-through to the global PIN); the global
    factors apply only to callers with no per-caller factor of that kind.
    """

    def __init__(self, config, store: VerificationStore):
        self.config = config
        self.store = store

    # -- resolution helpers ---------------------------------------------
    def _resolve_totp_secret(self, caller_id: str) -> Optional[str]:
        record = self.store.get(caller_id)
        secret = (record or {}).get("totp_secret")
        if secret:
            return secret
        return getattr(self.config, "verify_totp_secret", "") or None

    def is_configured(self) -> bool:
        """True when a *global* factor is set (applies to every caller)."""
        return bool(getattr(self.config, "verify_pin", "")
                    or getattr(self.config, "verify_totp_secret", ""))

    def has_any_credentials(self, caller_id: str) -> bool:
        record = self.store.get(caller_id) or {}
        return bool(record.get("pin_hash") or record.get("totp_secret"))

    def can_verify(self, caller_id: str) -> bool:
        """Whether verification is even possible for this caller."""
        return self.is_configured() or self.has_any_credentials(caller_id)

    # -- checks ----------------------------------------------------------
    def verify_pin(self, caller_id: str, candidate: str) -> bool:
        candidate = (candidate or "").strip()
        if not candidate:
            return False
        record = self.store.get(caller_id) or {}
        pin_hash = record.get("pin_hash")
        salt = record.get("pin_salt")
        if pin_hash and salt:
            try:
                computed = _hash_pin(candidate, bytes.fromhex(salt))
            except Exception:
                return False
            return _const_eq(computed, pin_hash)
        # No per-caller PIN — fall back to the global PIN if configured.
        global_pin = getattr(self.config, "verify_pin", "") or ""
        if not global_pin:
            return False
        return _const_eq(candidate, global_pin)

    def verify_totp(self, caller_id: str, candidate: str) -> bool:
        candidate = (candidate or "").strip()
        if not candidate or pyotp is None:
            return False
        secret = self._resolve_totp_secret(caller_id)
        if not secret:
            return False
        try:
            window = int(getattr(self.config, "verify_totp_window", 1))
            totp = build_totp(secret, totp_params(self.config))
            return bool(totp.verify(candidate, valid_window=window))
        except Exception as e:
            logger.warning(f"verify_totp failed for {caller_id}: {e}")
            return False

    def verify(self, caller_id: str, candidate: str,
               method: str = "auto") -> Tuple[bool, Optional[str]]:
        """Check a single entered code. Returns (ok, method_used).

        ``method``: "pin" or "otp" forces one factor; "auto" (default) accepts
        either — it tries OTP first when a TOTP secret is available, then the PIN.
        """
        method = (method or "auto").lower()
        if method == "pin":
            ok = self.verify_pin(caller_id, candidate)
            return ok, ("pin" if ok else None)
        if method == "otp":
            ok = self.verify_totp(caller_id, candidate)
            return ok, ("otp" if ok else None)
        # auto: prefer OTP when a secret exists, then PIN.
        if self._resolve_totp_secret(caller_id) and self.verify_totp(caller_id, candidate):
            return True, "otp"
        if self.verify_pin(caller_id, candidate):
            return True, "pin"
        return False, None

    def verify_explicit(self, candidate: str, pin: Optional[str] = None,
                        totp_secret: Optional[str] = None,
                        method: str = "auto",
                        totp_digits: Optional[int] = None,
                        totp_period: Optional[int] = None,
                        totp_algorithm: Optional[str] = None,
                        totp_window: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """Check a candidate code against explicitly-supplied factors.

        No store or global-config lookup: the PIN/secret come from the caller of
        this method (e.g. an n8n workflow that holds them itself). Same method
        semantics as ``verify`` — "pin"/"otp" force a factor, "auto" tries OTP
        first (when a secret is given) then the PIN. The TOTP digits/period/
        algorithm/window default to the configured values when not overridden.
        """
        candidate = (candidate or "").strip()
        if not candidate:
            return False, None

        def _pin_ok() -> bool:
            return bool(pin) and _const_eq(candidate, pin)

        def _otp_ok() -> bool:
            if not totp_secret or pyotp is None:
                return False
            try:
                window = (totp_window if totp_window is not None
                          else int(getattr(self.config, "verify_totp_window", 1)))
                params = totp_params(self.config, digits=totp_digits,
                                     period=totp_period, algorithm=totp_algorithm)
                return bool(build_totp(totp_secret, params).verify(
                    candidate, valid_window=window))
            except Exception as e:
                logger.warning(f"verify_explicit TOTP check failed: {e}")
                return False

        method = (method or "auto").lower()
        if method == "pin":
            return (True, "pin") if _pin_ok() else (False, None)
        if method == "otp":
            return (True, "otp") if _otp_ok() else (False, None)
        if totp_secret and _otp_ok():
            return True, "otp"
        if _pin_ok():
            return True, "pin"
        return False, None

    # -- async facades ---------------------------------------------------
    # PBKDF2 at 200k iterations takes tens of milliseconds; the checks run on
    # the same event loop that drives RTP/VAD/TTS for the live call, so the
    # call-side paths use these thread-offloaded variants.
    async def averify(self, caller_id: str, candidate: str,
                      method: str = "auto") -> Tuple[bool, Optional[str]]:
        return await asyncio.to_thread(self.verify, caller_id, candidate, method)

    async def averify_pin(self, caller_id: str, candidate: str) -> bool:
        return await asyncio.to_thread(self.verify_pin, caller_id, candidate)

    async def averify_totp(self, caller_id: str, candidate: str) -> bool:
        return await asyncio.to_thread(self.verify_totp, caller_id, candidate)

    async def averify_explicit(self, candidate: str, **kwargs) -> Tuple[bool, Optional[str]]:
        return await asyncio.to_thread(lambda: self.verify_explicit(candidate, **kwargs))

    def has_own_totp_secret(self, caller_id: str) -> bool:
        """True only when the caller has a per-caller (enrolled) TOTP secret."""
        record = self.store.get(caller_id) or {}
        return bool(record.get("totp_secret"))

    def current_otp(self, caller_id: str) -> Optional[Tuple[str, int]]:
        """(current_code, seconds_until_expiry) for the caller's secret, or None."""
        if pyotp is None:
            return None
        secret = self._resolve_totp_secret(caller_id)
        if not secret:
            return None
        try:
            params = totp_params(self.config)
            totp = build_totp(secret, params)
            # Code and remaining seconds come from ONE timestamp so they can't
            # straddle a step boundary and describe different codes.
            now = int(time.time())
            code = totp.at(now)
            period = int(params["period"])
            remaining = period - now % period
            return code, remaining
        except Exception as e:
            logger.warning(f"current_otp failed for {caller_id}: {e}")
            return None
