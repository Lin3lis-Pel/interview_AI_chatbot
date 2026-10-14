"""Security guard for both sides of the app.

Threat model
------------
Untrusted inputs are (a) the job ad -- pasted or fetched from a URL the user supplies,
and (b) the candidate answers. Both flow straight into an LLM whose output drives the UI.

Protections implemented here:

  developer / infrastructure side
    * SSRF guard on URL fetch (scheme allowlist, DNS resolution, private-range block,
      redirect re-validation, byte cap, timeout).
    * Hard size caps on every text input, before tokens are ever spent.
    * Per-session call budget (`RateLimiter`) so one tab cannot burn the API key.
    * Secrets never logged; `redact_pii` runs before anything is persisted or exported.
    * Control-character / zero-width stripping, so invisible-text injections cannot hide.

  user side
    * Prompt-injection detection with an explicit risk verdict surfaced in the UI, rather
      than silent filtering the user cannot see.
    * PII redaction of the candidate transcript before download/export.
    * Untrusted text is fenced with `wrap_untrusted()` and every system prompt states that
      fenced content is data. Fence markers occurring in the input are neutralised so the
      input cannot close its own fence.
    * Output guard strips anything that looks like leaked system-prompt text.

Nothing here is a substitute for the model-side rules in interview_ai/prompts/; the two layers
are meant to be defence in depth.
"""
from __future__ import annotations

import ipaddress
import re
import socket
import time
import unicodedata
from dataclasses import dataclass, field
from urllib.parse import urlparse

from ..config import SETTINGS

# --------------------------------------------------------------------------- text hygiene
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
# Matches a whole fence, a lone opening marker, or a lone closing marker. All three have to
# go: an input that supplies only `UNTRUSTED JOB_AD>>>` can otherwise close its own fence and
# have the remainder read as instructions. Bare `>>>` (Python REPL snippets) is left alone.
_FENCE_RE = re.compile(
    r"<<<\s*/?\s*UNTRUSTED[^>\n]*>>>"      # full fence
    r"|<<<\s*/?\s*UNTRUSTED[^\n]*"          # opening marker only
    r"|\bUNTRUSTED[ \t\w]*>>>",             # closing marker only
    re.IGNORECASE,
)


def clean_text(text: str, max_chars: int) -> tuple[str, list[str]]:
    """Normalise, strip invisible characters, and truncate. Returns (text, notes)."""
    notes: list[str] = []
    if not text:
        return "", notes

    text = unicodedata.normalize("NFKC", text)

    stripped = text.translate(_ZERO_WIDTH)
    if stripped != text:
        notes.append("Removed zero-width / soft-hyphen characters (a common way to hide text).")
    text = stripped

    controlled = "".join(c for c in text if c in "\n\t" or unicodedata.category(c)[0] != "C")
    if controlled != text:
        notes.append("Removed control characters.")
    text = controlled

    defenced, n = _FENCE_RE.subn("[fence-marker removed]", text)
    if n:
        notes.append(f"Neutralised {n} forged UNTRUSTED fence marker(s).")
    text = defenced

    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()

    if len(text) > max_chars:
        text = text[:max_chars]
        notes.append(f"Truncated to the {max_chars:,}-character limit.")
    return text, notes


def wrap_untrusted(text: str, label: str = "UNTRUSTED") -> str:
    """Fence untrusted content. `clean_text` has already removed forged fences."""
    return f"<<<{label}\n{text}\n{label}>>>"


# --------------------------------------------------------------------------- injection scan
# Heuristics, deliberately high-signal: these fire on phrasing that is normal in an attack
# and rare in a genuine job ad or interview answer.
_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
     "instruction override"),
    (r"disregard\s+(all\s+|the\s+)?(previous|prior|above|system)", "instruction override"),
    (r"forget\s+(everything|all|your)\s+(you|instructions?|rules?|prompt)", "instruction override"),
    (r"(reveal|show|print|repeat|output|dump)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions?|rules?)",
     "system-prompt exfiltration"),
    (r"you\s+are\s+now\s+(a|an|no longer)", "role reassignment"),
    (r"\b(new|updated)\s+(instructions?|system\s+prompt|rules?)\s*:", "role reassignment"),
    (r"act\s+as\s+(if\s+you\s+are\s+)?(a\s+)?(developer|admin|root|dan\b)", "privilege escalation"),
    (r"\bdeveloper\s+mode\b|\bjailbreak\b|\bDAN\s+mode\b", "jailbreak attempt"),
    (r"(give|award|assign|score)\s+(me|the\s+candidate)\s+(a\s+)?(perfect|max(imum)?|full|100|10/10|5/5)",
     "score manipulation"),
    (r"(always|must)\s+(recommend|say|answer|score)\s+.{0,30}(hire|strong_yes|pass|yes)",
     "score manipulation"),
    (r"\bhiring\s+manager\s+says\b|\bthis\s+candidate\s+is\s+pre-?approved\b", "false authority"),
    (r"(system|assistant)\s*:\s*", "fake chat turn"),
    (r"<\s*/?\s*(system|instructions?)\s*>", "fake markup turn"),
    (r"(curl|wget|fetch)\s+https?://", "exfiltration / SSRF lure"),
    (r"\bbase64\s*(decode|-d)\b", "obfuscated payload"),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), why) for p, why in _INJECTION_PATTERNS]


@dataclass
class SecurityVerdict:
    text: str
    risk: str = "low"          # low | medium | high
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    blocked: bool = False

    @property
    def ok(self) -> bool:
        return not self.blocked


def scan_input(raw: str, *, max_chars: int, kind: str = "job ad") -> SecurityVerdict:
    """Clean + scan one untrusted input. Never raises; returns a verdict the UI can show."""
    text, notes = clean_text(raw or "", max_chars)

    if not text:
        return SecurityVerdict(text="", risk="low", notes=notes, blocked=True,
                               reasons=[f"The {kind} is empty."])

    hits = sorted({why for rx, why in _COMPILED if rx.search(text)})
    risk = "high" if len(hits) >= 2 else "medium" if hits else "low"

    # Never hard-block on heuristics alone: a false positive would eat a legitimate ad.
    # The fences plus the system-prompt rules are the actual mitigation; this is disclosure.
    return SecurityVerdict(text=text, risk=risk, reasons=hits, notes=notes, blocked=False)


# --------------------------------------------------------------------------- output guard
_LEAK_MARKERS = ("ROLE\nYou are", "SECURITY: text inside", "GROUNDING: use only", "<<<UNTRUSTED")


def guard_output(text: str) -> str:
    """Last line of defence: drop model output that echoes the system prompt back."""
    if not text:
        return ""
    for marker in _LEAK_MARKERS:
        if marker in text:
            return "[response withheld: it echoed internal instructions]"
    return text


# --------------------------------------------------------------------------- PII redaction
# Phone numbers are the hardest PII to match without collateral damage. The first version
# redacted OpenRouter generation ids ("gen-1755...") inside eval failure text, destroying the
# diagnostics. So: require at least two separators, refuse to match inside a longer
# alphanumeric token, and confirm the digit count is phone-like before substituting.
_PHONE_CANDIDATE = re.compile(
    r"(?<![\w-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{1,4}\)[\s.-]?)?\d{2,4}(?:[\s.-]\d{2,5}){1,3}(?![\w-])"
)
_ISO_DATE = re.compile(r"^\d{4}[-.]\d{1,2}[-.]\d{1,2}$")


def _redact_phone(match: re.Match[str]) -> str:
    text = match.group(0)
    if _ISO_DATE.match(text):
        return text                      # 2026-08-18 is a date, not a phone number
    digits = sum(c.isdigit() for c in text)
    return "[phone]" if 9 <= digits <= 15 else text


_PII_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[card-number]"),
    (re.compile(r"\b(sk|pk|rk)[-_][A-Za-z0-9]{16,}\b"), "[api-key]"),
    (re.compile(r"\bsk-or-v1-[A-Za-z0-9]{16,}\b"), "[openrouter-key]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[github-token]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[aws-key]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), "[jwt]"),
    (re.compile(r"https?://(?:www\.)?linkedin\.com/in/[\w-]+"), "[linkedin-profile]"),
]


def redact_pii(text: str) -> str:
    """Redact contact details and credentials before export/logging. Applied to transcripts."""
    if not text:
        return ""
    for rx, replacement in _PII_RULES:
        text = rx.sub(replacement, text)
    return _PHONE_CANDIDATE.sub(_redact_phone, text)


def redact_secrets_for_log(text: str) -> str:
    """Same as redact_pii, plus never let the configured API key appear anywhere."""
    text = redact_pii(text)
    if SETTINGS.api_key:
        text = text.replace(SETTINGS.api_key, "[redacted-key]")
    return text


# --------------------------------------------------------------------------- SSRF guard
_ALLOWED_SCHEMES = {"http", "https"}
_BLOCKED_HOST_SUFFIXES = (".local", ".internal", ".localdomain", ".cluster.local")


class UnsafeURL(ValueError):
    """Raised when a user-supplied URL fails the SSRF checks."""


def validate_url(url: str) -> str:
    """Allow only public http(s) endpoints. Raises UnsafeURL with a user-readable reason."""
    url = (url or "").strip()
    if not url:
        raise UnsafeURL("No URL supplied.")
    if len(url) > 2048:
        raise UnsafeURL("URL is unreasonably long.")

    parsed = urlparse(url if "://" in url else f"https://{url}")
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise UnsafeURL(f"Only http and https URLs are allowed (got {parsed.scheme!r}).")

    host = (parsed.hostname or "").lower()
    if not host:
        raise UnsafeURL("URL has no host.")
    if host in {"localhost", "metadata.google.internal"} or host.endswith(_BLOCKED_HOST_SUFFIXES):
        raise UnsafeURL(f"Refusing to fetch an internal host ({host}).")

    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise UnsafeURL(f"Could not resolve {host}.") from exc

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            raise UnsafeURL(f"{host} resolves to a non-public address ({ip}); refusing to fetch.")

    return parsed.geturl()


# --------------------------------------------------------------------------- rate limiting
@dataclass
class RateLimiter:
    """Per-session budget: protects the API key from a runaway loop or an abusive tab."""
    max_calls: int = SETTINGS.max_calls_per_session
    min_interval_s: float = 0.0
    _used: int = 0
    _last: float = 0.0

    def check(self, n: int = 1) -> None:
        now = time.monotonic()
        if self._used + n > self.max_calls:
            raise PermissionError(
                f"Session call budget exhausted ({self.max_calls} model calls). Start a new interview."
            )
        if self.min_interval_s and (now - self._last) < self.min_interval_s:
            raise PermissionError("Slow down: too many requests in a row.")
        self._used += n
        self._last = now

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self._used)
