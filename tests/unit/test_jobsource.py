"""Job-ad ingestion: LinkedIn URL handling and HTML extraction."""
from __future__ import annotations

import pytest

from interview_ai.utils.jobsource import FetchError, _extract, candidate_urls, fetch_job_ad, linkedin_job_id


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.linkedin.com/jobs/view/4012345678", "4012345678"),
        ("https://www.linkedin.com/jobs/view/ai-engineer-at-acme-4012345678/", "4012345678"),
        ("https://www.linkedin.com/jobs/search/?currentJobId=4012345678&keywords=ai", "4012345678"),
        ("https://example.com/careers/ai-engineer", None),
    ],
)
def test_linkedin_job_id_extraction(url, expected):
    assert linkedin_job_id(url) == expected


def test_linkedin_urls_try_the_guest_endpoint_first():
    urls = candidate_urls("https://www.linkedin.com/jobs/view/4012345678")
    assert "jobs-guest" in urls[0]
    assert len(urls) == 2


def test_non_linkedin_urls_are_fetched_as_is():
    assert candidate_urls("https://example.com/jobs/1") == ["https://example.com/jobs/1"]


def test_html_extraction_drops_scripts_and_markup():
    html = """
    <html><head><style>.a{color:red}</style><script>alert('x')</script></head>
    <body><h1>Senior Data Engineer</h1>
    <p>You will build&nbsp;pipelines with Airflow and dbt.</p>
    <ul><li>5+ years SQL</li><li>Spark</li></ul></body></html>
    """
    text = _extract(html)
    assert "Senior Data Engineer" in text and "Airflow" in text
    assert "alert(" not in text and "color:red" not in text and "<" not in text


async def test_ssrf_target_is_refused_before_any_request():
    with pytest.raises(FetchError, match="non-public|internal"):
        await fetch_job_ad("http://169.254.169.254/latest/meta-data/")


async def test_unsupported_scheme_is_refused():
    with pytest.raises(FetchError, match="http and https"):
        await fetch_job_ad("file:///etc/passwd")
