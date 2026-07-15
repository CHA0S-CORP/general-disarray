"""Unit tests for the per-call From display-name override (caller_name).

`caller_name` lets an outbound call announce itself as e.g. "Weather Alert"
instead of the agent's extension, without disturbing the registered account
identity. The value lands inside a quoted string in a SIP From header, so it
has to be sanitized: a stray quote or angle bracket would terminate the
display name and corrupt the header.
"""
import pytest

from sip_handler import SIPHandler

pytestmark = pytest.mark.unit


@pytest.fixture
def handler(config_factory):
    cfg = config_factory(sip_user="42", sip_domain="pbx.local")
    return SIPHandler(cfg, on_call_callback=lambda call: None)


def test_display_name_wraps_the_account_uri(handler):
    assert handler._local_uri("Weather Alert") == '"Weather Alert" <sip:42@pbx.local>'


def test_empty_name_falls_back_to_the_bare_account_uri(handler):
    assert handler._local_uri("") == "sip:42@pbx.local"
    assert handler._local_uri("   ") == "sip:42@pbx.local"


@pytest.mark.parametrize("evil", [
    'Alert" <sip:attacker@evil.com>, "x',   # break out of the quoted string
    'Alert<sip:attacker@evil.com>',          # inject a second URI
    'Alert\\"',                              # escaped quote
    "Alert\r\nTo: victim@example.com",       # header injection via CRLF
])
def test_header_breaking_characters_are_stripped(handler, evil):
    uri = handler._local_uri(evil)
    # Exactly one quoted display name, one angle-bracketed URI, and that URI is
    # ours — nothing the caller supplied can redirect or extend the header.
    assert uri.count('"') == 2
    assert uri.count("<") == 1 and uri.count(">") == 1
    assert uri.endswith("<sip:42@pbx.local>")
    assert "\r" not in uri and "\n" not in uri
    assert "attacker" not in uri.split("<")[1]
