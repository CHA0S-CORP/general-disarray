"""End-to-end (real-stack) test harness. DGX/GPU only.

Brings up docker-compose.dgx.yml (real vLLM + Speaches + Redis + sip-agent),
drives REAL SIP calls through a containerized `pjsua` softphone, and asserts via
three layers:

  Layer 0  assert_spoke(wav)      - agent produced non-silent audio (RMS/duration)
  Layer 1  agent_events(since)    - structured JSON log events prove pipeline stages
  Layer 2  transcribe(wav)        - Speaches transcript keyword match (deterministic tools)

Env knobs:
  E2E_USE_RUNNING=1   run against an already-up stack (skip compose up/down)
  E2E_KEEP_STACK=1    bring up once, skip teardown (fast re-runs)
  E2E_COMPOSE_FILE    override compose file (default: <repo>/docker-compose.dgx.yml)
  E2E_NETWORK         compose network for the softphone (default: general-disarray_default)
  E2E_AGENT_API       default http://localhost:8080
  E2E_SPEACHES        default http://localhost:8001
  E2E_SIP_TARGET      default sip:ai-assistant@sip-agent:5060
  E2E_SOFTPHONE_IMAGE prebuilt softphone image tag (skips the build)
"""
import io
import json
import os
import shutil
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "audio"))
from gen_audio import ensure_question_wav  # noqa: E402

pytestmark = pytest.mark.e2e

# --- paths / config --------------------------------------------------------
_E2E_DIR = Path(__file__).resolve().parent
_AUDIO_DIR = _E2E_DIR / "audio"
_REPO_ROOT = _E2E_DIR.parents[2]  # tests/e2e -> tests -> sip-agent -> general-disarray

COMPOSE_FILE = os.environ.get("E2E_COMPOSE_FILE", str(_REPO_ROOT / "docker-compose.dgx.yml"))
USE_RUNNING = os.environ.get("E2E_USE_RUNNING", "0") == "1"
KEEP_STACK = os.environ.get("E2E_KEEP_STACK", "0") == "1"
NETWORK = os.environ.get("E2E_NETWORK", "general-disarray_default")
AGENT_API = os.environ.get("E2E_AGENT_API", "http://localhost:8080")
SPEACHES_URL = os.environ.get("E2E_SPEACHES", "http://localhost:8001")
SIP_TARGET = os.environ.get("E2E_SIP_TARGET", "sip:ai-assistant@sip-agent:5060")
SOFTPHONE_IMAGE_ENV = os.environ.get("E2E_SOFTPHONE_IMAGE")
_BUILT_IMAGE_TAG = "sipbot-test-softphone:latest"


def _require_docker():
    if shutil.which("docker") is None:
        pytest.skip("docker not available; e2e tier requires Docker + the DGX stack")


def _compose(*args, check=True, capture=True):
    return subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, *args],
        check=check, capture_output=capture, text=True,
    )


def _service_healthy(service: str) -> bool:
    """True if the compose service is healthy (or has no healthcheck but is up)."""
    out = _compose("ps", "--format", "json", check=False).stdout or ""
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    for r in rows:
        if r.get("Service") == service:
            health = r.get("Health", "")
            state = r.get("State", "")
            if health:
                return health == "healthy"
            return state == "running"
    return False


def _http_ok(url: str) -> bool:
    try:
        return httpx.get(url, timeout=5.0).status_code == 200
    except Exception:
        return False


def _wait(predicate, timeout: float, interval: float = 5.0, what: str = "condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if predicate():
                return
        except Exception:
            pass
        time.sleep(interval)
    raise TimeoutError(f"timed out after {timeout}s waiting for {what}")


# --- log scraping (Layer 1) ------------------------------------------------

def _agent_events(since: datetime | None = None):
    """Return structured JSON log events emitted by the sip-agent service.

    The app's JSONFormatter writes one JSON object per line; uvicorn/other lines
    are skipped. Pass `since` to bound the window to the current call.
    """
    args = ["logs", "--no-color", "--no-log-prefix"]
    if since is not None:
        args += ["--since", since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")]
    args += ["sip-agent"]
    out = _compose(*args, check=False).stdout or ""
    events = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _event_names(events):
    # JSONFormatter nests the event tag under "event" (top-level or in "data").
    names = []
    for e in events:
        name = e.get("event") or (e.get("data") or {}).get("event")
        if name:
            names.append(name)
    return names


# --- audio assertions (Layers 0 & 2) ---------------------------------------

def _wav_rms_and_duration(path: Path):
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
    if samples.size == 0:
        return 0.0, 0.0
    rms = float(np.sqrt(np.mean(samples ** 2)))
    duration = samples.size / rate
    return rms, duration


def _transcribe(path: Path) -> str:
    with open(path, "rb") as f:
        files = {"file": ("captured.wav", f, "audio/wav")}
        data = {"model": "Systran/faster-distil-whisper-small.en", "response_format": "json"}
        resp = httpx.post(
            f"{SPEACHES_URL.rstrip('/')}/v1/audio/transcriptions",
            files=files, data=data, timeout=120.0,
        )
    resp.raise_for_status()
    return (resp.json().get("text") or "").strip()


# --- fixtures --------------------------------------------------------------

@pytest.fixture(scope="session")
def stack():
    """Bring up the full stack (unless reusing a running one); wait for health."""
    _require_docker()
    brought_up = False
    if not USE_RUNNING:
        _compose("up", "-d")
        brought_up = True

    # vLLM dominates cold-start (start_period 300s); be generous.
    _wait(lambda: _service_healthy("redis"), timeout=90, what="redis healthy")
    _wait(lambda: _service_healthy("speaches"), timeout=480, what="speaches healthy")
    _wait(lambda: _service_healthy("vllm"), timeout=900, what="vllm healthy")
    # sip-agent has no compose healthcheck: poll /health AND wait for the
    # `ready` log event (TTS precache done -> Speaches path warmed).
    _wait(lambda: _http_ok(f"{AGENT_API}/health"), timeout=240, what="agent /health")
    _wait(lambda: "ready" in _event_names(_agent_events()), timeout=240, what="agent ready event")

    yield {"agent_api": AGENT_API, "speaches": SPEACHES_URL}

    if brought_up and not KEEP_STACK:
        _compose("down", check=False)


@pytest.fixture(scope="session")
def softphone_image(stack):
    """Build (once) the pjsua softphone image, unless a tag is provided."""
    if SOFTPHONE_IMAGE_ENV:
        return SOFTPHONE_IMAGE_ENV
    subprocess.run(
        ["docker", "build", "-f", str(_E2E_DIR / "Dockerfile.softphone"),
         "-t", _BUILT_IMAGE_TAG, str(_E2E_DIR)],
        check=True,
    )
    return _BUILT_IMAGE_TAG


@pytest.fixture
def question_wav(stack):
    """Factory: ensure a silence-padded question WAV exists; return its filename
    (relative to the shared audio dir, which is mounted into the softphone)."""
    def _make(filename: str) -> str:
        ensure_question_wav(filename, SPEACHES_URL, _AUDIO_DIR)
        return filename
    return _make


@pytest.fixture
def place_inbound_call(softphone_image):
    """Dial the agent, play a question WAV, record the reply. Returns (captured_wav,
    call_started_at). The softphone shares the compose network and mounts the
    audio dir at /audio."""
    def _call(question_filename: str, duration: int = 30, capture_name: str = "captured.wav"):
        captured = _AUDIO_DIR / capture_name
        if captured.exists():
            captured.unlink()
        cname = "e2e-dialer"
        subprocess.run(["docker", "rm", "-f", cname], check=False, capture_output=True)
        started_at = datetime.now(timezone.utc)
        # pjsua reads commands from stdin and quits on EOF. We must keep stdin
        # OPEN during the call (else it cancels instantly), then CLOSE it after
        # the call window so pjsua quits cleanly — which finalizes the WAV
        # recorder and exits the container. `--duration` hangs up the call but
        # does NOT quit pjsua, so closing stdin is what actually ends it.
        # stdout/stderr are discarded (an unread PIPE would fill and stall pjsua).
        proc = subprocess.Popen(
            [
                "docker", "run", "--rm", "-i", "--name", cname,
                "--network", NETWORK,
                "-v", f"{_AUDIO_DIR}:/audio",
                softphone_image,
                "--rtp-port", "4000",
                "--auto-play", "--play-file", f"/audio/{question_filename}",
                "--auto-rec", "--rec-file", f"/audio/{capture_name}",
                "--duration", str(duration),
                "--stdout-no-buf",
                SIP_TARGET,
            ],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(duration + 4)  # let the call run; it hangs up at --duration
        try:
            if proc.stdin:
                proc.stdin.close()  # EOF -> pjsua quits, finalizes WAV, container exits
        except Exception:
            pass
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        subprocess.run(["docker", "rm", "-f", cname], check=False, capture_output=True)
        return captured, started_at
    return _call


@pytest.fixture
def agent_events():
    """Expose the Layer-1 log-event reader to tests."""
    def _read(since: datetime | None = None):
        return _agent_events(since)
    return _read


@pytest.fixture
def event_names():
    return _event_names


@pytest.fixture
def assert_spoke():
    """Layer 0: assert the captured WAV is real, non-silent agent speech."""
    def _assert(path: Path, min_duration_s: float = 1.5, min_rms: float = 150.0):
        assert path.exists(), f"no captured audio at {path}"
        rms, duration = _wav_rms_and_duration(path)
        assert duration >= min_duration_s, f"captured audio too short: {duration:.2f}s"
        assert rms >= min_rms, f"captured audio is silent (rms={rms:.1f})"
        return rms, duration
    return _assert


@pytest.fixture
def transcribe():
    """Layer 2: transcribe a WAV via the stack's Speaches."""
    return _transcribe


@pytest.fixture
def agent_post(stack):
    """POST to the agent's REST API (returns the httpx.Response, never raises)."""
    def _post(path: str, json_body: dict, timeout: float = 30.0):
        return httpx.post(f"{AGENT_API}{path}", json=json_body, timeout=timeout)
    return _post


class _Answerer:
    def __init__(self, name, captured, proc):
        self.name = name
        self.captured = captured
        self.proc = proc

    def stop_and_finalize(self):
        """Close pjsua's stdin (EOF) so it quits cleanly, finalizing the WAV
        recorder and exiting the container — same mechanism as the dialer."""
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        subprocess.run(["docker", "rm", "-f", self.name], check=False, capture_output=True)


@pytest.fixture
def outbound_answerer(softphone_image):
    """Resident auto-answer softphone reachable on the compose network as
    `test-softphone:5060`. The agent dials it for outbound-call tests.

    Runs as a non-blocking Popen with a held-open stdin (pjsua quits on EOF), so
    `stop_and_finalize()` can close stdin for a clean shutdown that finalizes the
    recording — exactly the pattern the inbound dialer uses."""
    name = "test-softphone"
    captured = _AUDIO_DIR / "outbound_captured.wav"
    if captured.exists():
        captured.unlink()
    subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)
    proc = subprocess.Popen(
        [
            "docker", "run", "--rm", "-i", "--name", name,
            "--network", NETWORK,
            "-v", f"{_AUDIO_DIR}:/audio",
            softphone_image,
            "--id", "sip:tester@test-softphone", "--local-port", "5060",
            "--rtp-port", "4000",
            "--auto-answer", "200",
            "--auto-rec", "--rec-file", "/audio/outbound_captured.wav",
            "--duration", "120",
        ],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(3)  # let pjsua come up and start listening before the agent dials
    answerer = _Answerer(name, captured, proc)
    try:
        yield answerer
    finally:
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)


@pytest.fixture
def wait_for_event():
    """Poll the agent log until `name` appears after `since` (or time out)."""
    def _wait_for(name: str, since, timeout: float = 60.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if name in _event_names(_agent_events(since)):
                return True
            time.sleep(2.0)
        return False
    return _wait_for
