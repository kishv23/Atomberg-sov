from sov.collectors.google import fetch_google
from sov.collectors.youtube import fetch_youtube
from sov.expand import expand_query, plan_crawl


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        return self.responses.pop(0)


def page(n, offset=0):
    return {"items": [{"link": f"https://x/{offset + i}", "title": f"t{offset + i}", "snippet": "s", "displayLink": "x"} for i in range(n)]}


def test_google_paginates_and_passes_query_as_param():
    session = FakeSession([FakeResponse(200, page(10)), FakeResponse(200, page(5, 10)), FakeResponse(200, {})])
    rows = fetch_google("k", "cx", "smart fan & more", session=session, sleep=lambda s: None)
    assert len(rows) == 15
    assert rows[-1]["rank"] == 15
    assert session.calls[0]["q"] == "smart fan & more"  # encoded by requests, not concatenated
    assert session.calls[1]["start"] == 11


def test_google_retries_on_rate_limit():
    slept = []
    session = FakeSession([FakeResponse(429, {}), FakeResponse(200, page(3)), FakeResponse(200, {})])
    rows = fetch_google("k", "cx", "q", session=session, sleep=slept.append)
    assert len(rows) == 3 and slept == [1]


def test_google_caps_at_100():
    session = FakeSession([FakeResponse(200, page(10, i * 10)) for i in range(12)])
    assert len(fetch_google("k", "cx", "q", max_results=500, session=session, sleep=lambda s: None)) == 100


class FakeYouTube:
    """Mimics googleapiclient's chained .search().list(...).execute() calls."""

    def __init__(self, pages):
        self.pages = pages

    def search(self):
        outer = self

        class _S:
            def list(self, pageToken=None, maxResults=50, **kw):
                idx = int(pageToken or 0)
                body = outer.pages[idx]
                return type("R", (), {"execute": lambda self: body})()

        return _S()

    def videos(self):
        class _V:
            def list(self, id, **kw):
                items = [{"id": v, "statistics": {"viewCount": "100", "likeCount": "5"}} for v in id.split(",")]
                return type("R", (), {"execute": lambda self: {"items": items}})()

        return _V()


def yt_item(vid):
    return {"id": {"videoId": vid}, "snippet": {"title": vid, "description": "d", "channelTitle": "c", "publishedAt": "2025-01-01T00:00:00Z"}}


def test_youtube_pages_dedupes_and_reads_stats():
    pages = [
        {"items": [yt_item("a"), yt_item("b")], "nextPageToken": "1"},
        {"items": [yt_item("b"), yt_item("c")]},
    ]
    rows = fetch_youtube(None, "smart fan", 10, client=FakeYouTube(pages))
    assert [r["title"] for r in rows] == ["a", "b", "c"]
    assert rows[0]["views"] == 100 and rows[0]["comments"] is None


def test_expand_query_dedupes_and_limits():
    variants = expand_query("smart fan", ["best", "review", "Best", ""], ["Atomberg"], limit=10)
    assert variants == ["smart fan", "best smart fan", "smart fan review", "Atomberg smart fan"]


def test_plan_crawl_respects_budget():
    plan = plan_crawl(["a", "b", "c", "d", "e", "f", "g"], budget=2000, youtube_per_variant=200, google_per_variant=100)
    assert sum(p["youtube"] + p["google"] for p in plan) == 2000
    assert plan[0] == {"query": "a", "youtube": 200, "google": 100}
