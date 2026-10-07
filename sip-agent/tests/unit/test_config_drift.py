"""Config drift guard: every env var config.py reads must be documented in the
root .env.example (commented-out entries count), so new settings don't ship
undiscoverable. Internal-only knobs go in the exemption list, with a reason.
"""
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_AGENT_DIR = Path(__file__).resolve().parents[2]
_CONFIG_PY = _AGENT_DIR / "src" / "config.py"
_ENV_EXAMPLE = _AGENT_DIR.parent / ".env.example"

# Env vars config.py reads that are deliberately NOT in .env.example.
EXEMPT = {
    # Container-internal path of the docker socket for CONTAINER_CTL; it must
    # match the (commented) volume mount in the compose files, not a user knob.
    "DOCKER_SOCKET_PATH",
}

_READ_RE = re.compile(
    r'(?:os\.getenv|os\.environ\.get|_env_bool|_load_phrases_from_env_or_default)'
    r'\(\s*"([A-Z][A-Z0-9_]*)"'
)


def _config_env_names():
    return set(_READ_RE.findall(_CONFIG_PY.read_text()))


def _documented(name: str, text: str) -> bool:
    # `NAME=...` or `# NAME=...` at the start of a line.
    return re.search(rf"(?m)^#?\s*{re.escape(name)}=", text) is not None


def test_parser_finds_config_vars():
    names = _config_env_names()
    # Sanity: the regex must keep matching the config.py idioms.
    assert {"SIP_USER", "KNOWLEDGE_ENABLED", "PHRASES_GREETINGS"} <= names
    assert len(names) > 100


@pytest.mark.skipif(not _ENV_EXAMPLE.exists(), reason=".env.example not present")
def test_every_config_var_documented_in_env_example():
    text = _ENV_EXAMPLE.read_text()
    missing = sorted(n for n in _config_env_names() - EXEMPT if not _documented(n, text))
    assert not missing, (
        f"config.py reads env vars not documented in .env.example: {missing}. "
        "Add them (commented-out is fine) or, if internal-only, to EXEMPT with a reason."
    )


def test_exemptions_are_still_read():
    """Stale exemptions hide nothing useful; drop them when config stops reading the var."""
    assert EXEMPT <= _config_env_names()
