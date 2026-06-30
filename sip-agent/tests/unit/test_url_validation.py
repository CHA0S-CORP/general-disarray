"""Unit tests for the outbound dial-target and webhook SSRF guards (api.py).

Uses literal IPs for the SSRF cases so no DNS/network is required to run.
"""
import pytest

from api import validate_extension, validate_callback_url, RequestRejected

pytestmark = pytest.mark.unit


# --- validate_extension (sync) ---------------------------------------------

def test_bare_extension_allowed(config):
    # Default config: outbound_allow_sip_uri is False.
    validate_extension("1001", config)  # must not raise


@pytest.mark.parametrize("ext", ["sip:tester@host", "tester@host", "", "   "])
def test_rejects_sip_uris_and_empty(config, ext):
    with pytest.raises(RequestRejected):
        validate_extension(ext, config)


def test_sip_uri_allowed_when_flag_set(config_factory):
    cfg = config_factory(outbound_allow_sip_uri="true")
    validate_extension("sip:tester@test-softphone:5060", cfg)  # must not raise


def test_extension_pattern_restriction(config_factory):
    cfg = config_factory(outbound_extension_pattern=r"\d{4}")
    validate_extension("1001", cfg)  # matches
    with pytest.raises(RequestRejected):
        validate_extension("12", cfg)  # too short for the pattern


# --- validate_callback_url (async) -----------------------------------------

async def test_none_url_is_noop(config):
    await validate_callback_url(None, config)  # must not raise


@pytest.mark.parametrize("url", ["http://127.0.0.1/hook", "http://10.0.0.1/hook"])
async def test_blocks_private_addresses(config, url):
    with pytest.raises(RequestRejected):
        await validate_callback_url(url, config)


async def test_blocks_bad_scheme(config):
    with pytest.raises(RequestRejected):
        await validate_callback_url("ftp://example.com/hook", config)


async def test_requires_https_when_configured(config_factory):
    cfg = config_factory(webhook_require_https="true", webhook_allow_private="true")
    with pytest.raises(RequestRejected):
        await validate_callback_url("http://10.0.0.1/hook", cfg)


async def test_private_allowed_when_flag_set(config_factory):
    cfg = config_factory(webhook_allow_private="true")
    await validate_callback_url("http://127.0.0.1/hook", cfg)  # must not raise
