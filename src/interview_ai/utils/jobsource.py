"""Get job-ad text from either a pasted blob or a URL.

LinkedIn serves most job pages behind auth and aggressive bot detection, so the URL path
is best-effort by design: when extraction fails or comes back too thin, we say so and ask
the user to paste instead, rather than feeding the analyser a cookie banner.
"""
from __future__ import annotations

import asyncio
import re

import httpx

from ..config import SETTINGS
from .security import UnsafeURL, validate_url

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
_MIN_USEFUL_CHARS = 400
_LOGIN_WALL = re.compile(
    r"(sign in to see|join linkedin|authwall|please enable javascript|"
    r"you must be logged in|access to this page has been denied)",
    re.IGNORECASE,
)


class FetchError(RuntimeError):
    """URL fetch or extraction failed in a way the user needs to know about."""


def linkedin_job_id(url: str) -> str | None:
    """LinkedIn job ids appear either in the path or as ?currentJobId=."""
    for pattern in (r"/jobs/view/(?:[\w-]*-)?(\d{6,})", r"currentJobId=(\d{6,})"):
        if m := re.search(pattern, url):
            return m.group(1)
    return None


def candidate_urls(url: str) -> list[str]:
    """LinkedIn's guest endpoint is server-rendered and usually readable without auth."""
    urls = [url]
    if "linkedin.com" in url and (job_id := linkedin_job_id(url)):
        urls.insert(
            0,
            f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}",
        )
    return urls


def _extract(html: str) -> str:
    """Prefer trafilatura; fall back to a tag strip so a missing dep is never fatal."""
    try:
        import trafilatura

        text = trafilatura.extract(
            html, include_comments=False, include_tables=True, favor_recall=True
        )
        if text:
            return text.strip()
    except Exception:  # noqa: BLE001 - extraction is best-effort by design
        pass

    text = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</div>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


async def fetch_job_ad(url: str) -> str:
    """Validate (SSRF guard), fetch with caps, extract readable text.

    Raises FetchError with a message written for the end user.
    """
    try:
        safe = validate_url(url)
    except UnsafeURL as exc:
        raise FetchError(str(exc)) from exc

    last_problem = "the page returned no readable text"
    limits = httpx.Limits(max_connections=4)

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=SETTINGS.url_fetch_timeout_s,
        limits=limits,
        headers={"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"},
    ) as client:
        for target in candidate_urls(safe):
            try:
                # Re-validate after redirects: the first hop can point anywhere.
                resp = await client.get(target)
                if str(resp.url) != target:
                    validate_url(str(resp.url))
            except (httpx.HTTPError, UnsafeURL, asyncio.TimeoutError) as exc:
                last_problem = f"request failed ({type(exc).__name__})"
                continue

            if resp.status_code >= 400:
                last_problem = f"the site answered HTTP {resp.status_code}"
                continue
            if len(resp.content) > SETTINGS.url_max_bytes:
                raise FetchError("That page is larger than the 2 MB fetch limit.")

            text = _extract(resp.text)
            if _LOGIN_WALL.search(text[:2000]) or _LOGIN_WALL.search(resp.text[:4000]):
                last_problem = "the page is behind a login wall"
                continue
            if len(text) >= _MIN_USEFUL_CHARS:
                return text
            last_problem = "the extracted text was too short to be a job ad"

    raise FetchError(
        f"Could not read that job ad -- {last_problem}. "
        "LinkedIn blocks most automated reads; open the posting and paste the text instead."
    )
