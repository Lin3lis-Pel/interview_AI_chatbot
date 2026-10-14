"""Cross-cutting helpers: input safety, job-ad fetching, prompt-text formatting.

`security` is the one every stage touches -- it scans and fences untrusted input, redacts
PII on the way out, and blocks SSRF on URL fetches. See docs/architecture.md.
"""
from __future__ import annotations
