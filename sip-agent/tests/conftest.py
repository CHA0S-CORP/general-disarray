"""Shared pytest fixtures and import-time setup for the SIP agent test suite.

This file runs before any test module is collected, so it is the right place to
pin environment variables that must be set *before* `config.Config` (and anything
that constructs it) is imported.
"""
import os
import sys
from pathlib import Path

# --- Import-time environment hardening -------------------------------------
# Keep telemetry a no-op so importing app modules never needs the OTel SDK.
os.environ.setdefault("OTEL_ENABLED", "false")

# Redirect Config's data_dir (recordings/, logs/) into a throwaway directory so
# tests never write into the repo. Config.__post_init__ mkdir's this on init.
_TEST_DATA_DIR = Path(__file__).parent / ".test-data"
os.environ.setdefault("DATA_DIR", str(_TEST_DATA_DIR))

# Make `src/` importable even when pytest is invoked oddly (pytest.ini also sets
# pythonpath=src, but belt-and-suspenders keeps `python -m pytest` from any cwd
# working and lets editors resolve imports).
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest  # noqa: E402


def make_config(**overrides):
    """Build a fresh `Config`, applying env-var overrides for this call only.

    Config reads every setting from `os.getenv` at construction time, so we set
    the requested env vars, build the object, then restore the prior environment.
    Pass plain values (e.g. ``stt_mode="realtime"`` -> sets ``STT_MODE``).
    """
    from config import Config

    env_map = {
        "speaches_api_url": "SPEACHES_API_URL",
        "llm_base_url": "LLM_BASE_URL",
        "llm_backend": "LLM_BACKEND",
        "llm_model": "LLM_MODEL",
        "llm_temperature": "LLM_TEMPERATURE",
        "stt_mode": "STT_MODE",
        "api_auth_token": "API_AUTH_TOKEN",
        "outbound_allow_sip_uri": "OUTBOUND_ALLOW_SIP_URI",
        "outbound_extension_pattern": "OUTBOUND_EXTENSION_PATTERN",
        "webhook_allow_private": "WEBHOOK_ALLOW_PRIVATE",
        "webhook_require_https": "WEBHOOK_REQUIRE_HTTPS",
        "data_dir": "DATA_DIR",
    }
    saved = {}
    try:
        for key, value in overrides.items():
            env_key = env_map.get(key, key.upper())
            saved[env_key] = os.environ.get(env_key)
            os.environ[env_key] = str(value)
        return Config()
    finally:
        for env_key, prev in saved.items():
            if prev is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = prev


@pytest.fixture
def config():
    """A default `Config` instance for tests that just need one."""
    return make_config()


@pytest.fixture
def config_factory():
    """Expose `make_config` as a fixture for tests needing overrides."""
    return make_config
