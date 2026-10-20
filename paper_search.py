"""Search providers for one PaSa paper-research run."""
import asyncio
import json
import os
import re
import uuid
import warnings
from datetime import datetime
from urllib.parse import urlsplit

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "pasa (+https://github.com/bytedance/pasa)"


class PaperSearch:
    def __init__(self, google_search, google_key, provider=None, timeout=30):
        provider = provider if provider is not None else os.getenv("PASA_SEARCH_PROVIDER", "auto")
        configured = bool(google_key and google_key.strip() and google_key != "your google keys")
        if provider == "auto":
            provider = "google" if configured else "parallel"
        if provider not in ("parallel", "google", "none"):
            raise ValueError("Search provider must be auto, parallel, google or none")
        if provider == "google" and not configured:
            raise ValueError("Google search requires SERPER_API_KEY or utils.GOOGLE_KEY")
        if timeout <= 0:
            raise ValueError("Search timeout must be positive")
        self.provider = provider
        self.google_search = google_search
        self.timeout = timeout
        # A PaperAgent owns this object for its whole research run, across threads
        # and follow-up searches. The HTTP connection is deliberately independent.
        self.session_id = str(uuid.uuid4())

    def search(self, query, num=10, end_date=None):
        if self.provider == "none" or num <= 0:
            return []
        if self.provider == "google":
            return self.google_search(query, num, end_date)
        try:
            return asyncio.run(asyncio.wait_for(self._parallel_search(query, num, end_date), self.timeout))
        except Exception as exc:
            # Preserve the result-list contract without silently switching providers.
            # Cancellation/KeyboardInterrupt are BaseExceptions and propagate.
            warnings.warn(f"Parallel search failed ({type(exc).__name__}): {exc}")
            return []

    async def _parallel_search(self, query, num, end_date):
        search_query = f"{query} site:arxiv.org"
        if end_date:
            try:
                date = datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d")
                search_query = f"{query} before:{date} site:arxiv.org"
            except ValueError:
                pass
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            timeout=self.timeout,
        ) as client:
            async with streamable_http_client(MCP_URL, http_client=client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("web_search", {
                        "objective": f"Find arXiv papers relevant to: {query}",
                        "search_queries": [search_query],
                        "session_id": self.session_id,
                    })
        if result.isError:
            message = " ".join(c.text for c in result.content if c.type == "text")
            raise ValueError(message or "MCP search returned an error")
        data = result.structuredContent
        if data is None:
            text = next(c.text for c in result.content if c.type == "text")
            data = json.loads(text)
        results = data["results"]
        if not isinstance(results, list):
            raise ValueError("MCP search results must be a list")
        ids = []
        for item in results:
            url = urlsplit(item.get("url", ""))
            if url.scheme not in ("http", "https") or url.hostname not in ("arxiv.org", "www.arxiv.org"):
                continue
            match = re.fullmatch(r"/(?:abs|pdf|html)/(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?/?", url.path)
            if match and match[1] not in ids:
                ids.append(match[1])
            if len(ids) >= num:
                break
        return ids
