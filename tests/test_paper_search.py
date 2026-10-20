"""Exercise the actual PaperAgent dispatcher with scripted models and MCP HTTP."""
import asyncio
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import httpx

import paper_search


ROOT = Path(__file__).resolve().parents[1]
PAPER_ID = "2501.10120"
PAPER = {
    "title": "PaSa", "abstract": "An agent for paper search.", "sections": {},
}


class MCPServer:
    """HTTP fixture beneath the real MCP SDK; retain every outgoing request."""
    def __init__(self, mode="structured", urls=None):
        self.mode = mode
        self.urls = urls if urls is not None else [f"https://arxiv.org/abs/{PAPER_ID}v2"]
        self.requests = []
        self.lock = threading.Lock()
        self.started = None

    async def handle(self, request):
        body = json.loads(request.content) if request.content else {}
        with self.lock:
            self.requests.append((request, body))
        method = body.get("method")
        if request.method in ("GET", "DELETE"):
            return httpx.Response(405)
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]}
        elif method == "tools/call":
            if self.started is not None:
                self.started.set()
            if self.mode == "slow":
                await asyncio.sleep(5)
            if self.mode == "http_error":
                return httpx.Response(503, text="unavailable")
            if self.mode == "transport_timeout":
                raise httpx.ReadTimeout("fixture timeout", request=request)
            if self.mode == "tool_error":
                result = {"isError": True, "content": [{"type": "text", "text": "quota exceeded"}]}
            else:
                data = {"results": [{"url": u} for u in self.urls]}
                if self.mode == "malformed":
                    data = {"results": None}
                result = {"content": [{"type": "text", "text": json.dumps(data)}]}
                if self.mode != "text":
                    result["structuredContent"] = data
        else:
            raise AssertionError(f"Unexpected MCP method: {method}")
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    def client(self, **kwargs):
        return REAL_HTTP_CLIENT(transport=httpx.MockTransport(self.handle), **kwargs)

    def calls(self):
        return [b["params"] for _, b in self.requests if b.get("method") == "tools/call"]


REAL_HTTP_CLIENT = httpx.AsyncClient


class Crawler:
    def infer(self, prompt):
        return "[Search]paper search agents[Stop][Search]academic research tools[Stop]"


class Selector:
    def __init__(self):
        self.prompts = []

    def infer_score(self, prompts):
        self.prompts.extend(prompts)
        return [0.9] * len(prompts)


class PaperSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use real utils metadata loading and zip lookup, without a model download.
        cls.tmp = tempfile.TemporaryDirectory()
        folder = Path(cls.tmp.name) / "data/paper_database"
        folder.mkdir(parents=True)
        (folder / "id2paper.json").write_text(json.dumps({PAPER_ID: PAPER["title"]}))
        with zipfile.ZipFile(folder / "cs_paper_2nd.zip", "w") as db:
            db.writestr("pasa", json.dumps(PAPER))
        previous = os.getcwd()
        try:
            os.chdir(cls.tmp.name)
            with patch.dict(os.environ, {}, clear=True):
                cls.agent_module = importlib.import_module("paper_agent")
                cls.utils = importlib.import_module("utils")
        finally:
            os.chdir(previous)

    @classmethod
    def tearDownClass(cls):
        cls.utils.paper_db.close()
        cls.tmp.cleanup()

    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def agent(self, provider=None):
        return self.agent_module.PaperAgent(
            "Find paper search agents", Crawler(), Selector(), end_date="20250120",
            prompts_path=str(ROOT / "agent_prompt.json"), expand_layers=0,
            search_provider=provider,
        )

    def test_default_dispatch_consumes_mcp_urls_metadata_and_selection(self):
        server = MCPServer()
        with patch.object(paper_search.httpx, "AsyncClient", server.client):
            agent = self.agent()
            self.assertEqual(agent.search_tool.provider, "parallel")
            agent.run()
        self.assertEqual(agent.root.extra["recall_papers"], ["PaSa"])
        self.assertEqual(agent.papers_queue[0].arxiv_id, PAPER_ID)
        self.assertIn("An agent for paper search.", agent.selector.prompts[0])
        self.assertEqual(len(server.calls()), 2)
        for call in server.calls():
            self.assertEqual(call["name"], "web_search")
            self.assertIn("before:2025-01-20 site:arxiv.org", call["arguments"]["search_queries"][0])
            self.assertEqual(call["arguments"]["session_id"], agent.search_tool.session_id)
        self.assertTrue(any(b.get("method") == "initialize" for _, b in server.requests))
        self.assertTrue(any(b.get("method") == "tools/list" for _, b in server.requests))
        for request, _ in server.requests:
            self.assertEqual(str(request.url), paper_search.MCP_URL)
            self.assertEqual(request.headers["user-agent"], paper_search.USER_AGENT)
            self.assertNotIn("authorization", request.headers)

    def test_preserves_configured_google_and_passes_its_arguments(self):
        callback = Mock(return_value=[PAPER_ID])
        with patch.object(self.utils, "GOOGLE_KEY", "configured-key"), \
                patch.object(self.agent_module, "google_search_arxiv_id", callback), \
                patch.object(paper_search.httpx, "AsyncClient") as mcp:
            agent = self.agent()
            agent.run()
            self.assertEqual(agent.search_tool.provider, "google")
            self.assertEqual(agent.root.extra["recall_papers"], ["PaSa"])
            self.assertEqual(callback.call_args.args[1:], (10, "20250120"))
            mcp.assert_not_called()

    def test_configured_google_failure_does_not_switch_to_parallel(self):
        callback = Mock(side_effect=RuntimeError("configured provider failed"))
        with patch.object(self.utils, "GOOGLE_KEY", "configured-key"), \
                patch.object(self.agent_module, "google_search_arxiv_id", callback), \
                patch.object(paper_search.httpx, "AsyncClient") as client:
            with self.assertRaisesRegex(RuntimeError, "configured provider failed"):
                self.agent().search_paper(["query"])
            client.assert_not_called()

    def test_cli_choice_overrides_environment_through_real_entrypoint(self):
        class ScriptedModel(Crawler, Selector):
            def __init__(self, model_path):
                Selector.__init__(self)
        input_file = Path(self.tmp.name) / "input.jsonl"
        input_file.write_text(json.dumps({"question": "paper agents", "source_meta": {"published_time": "20250120"}}) + "\n")
        argv = ["run_paper_agent.py", "--input_file", str(input_file), "--output_folder", "",
                "--expand_layers", "0", "--search_provider", "none"]
        with patch.dict(sys.modules, {"models": SimpleNamespace(Agent=ScriptedModel)}), \
                patch.dict(os.environ, {"PASA_SEARCH_PROVIDER": "parallel"}), \
                patch.object(sys, "argv", argv), \
                patch.object(paper_search.httpx, "AsyncClient") as client:
            result = runpy.run_path(str(ROOT / "run_paper_agent.py"), run_name="__main__")
            self.assertEqual(result["paper_agent"].search_tool.provider, "none")
            self.assertEqual(result["paper_agent"].papers_queue, [])
            client.assert_not_called()

    def test_environment_key_reaches_real_google_request(self):
        with patch.dict(os.environ, {"SERPER_API_KEY": "fixture-google-key"}), \
                patch.object(self.utils.requests, "request") as request:
            previous = os.getcwd()
            try:
                self.utils.paper_db.close()
                os.chdir(self.tmp.name)
                importlib.reload(self.utils)
                with patch.object(self.utils, "GOOGLE_KEY", self.utils.GOOGLE_KEY):
                    response = request.return_value
                    response.status_code = 200
                    response.text = json.dumps({"organic": [{"link": f"https://arxiv.org/abs/{PAPER_ID}"}]})
                    agent = self.agent()
                    agent.run()
                self.assertEqual(agent.search_tool.provider, "google")
                self.assertEqual(request.call_args.kwargs["headers"]["X-API-KEY"], "fixture-google-key")
                self.assertEqual(agent.root.extra["recall_papers"], ["PaSa"])
            finally:
                os.chdir(previous)
                self.utils.GOOGLE_KEY = "your google keys"

    def test_explicit_and_environment_choices_disable_and_invalid_config(self):
        server = MCPServer()
        with patch.dict(os.environ, {"PASA_SEARCH_PROVIDER": "none"}), \
                patch.object(self.utils, "GOOGLE_KEY", "configured-key"), \
                patch.object(paper_search.httpx, "AsyncClient", server.client):
            disabled = self.agent()
            disabled.run()
            self.assertEqual(disabled.search_tool.provider, "none")
            self.assertFalse(server.requests)
            explicit = self.agent("parallel")
            explicit.run()
            self.assertEqual(explicit.root.extra["recall_papers"], ["PaSa"])
            self.assertEqual(self.agent("auto").search_tool.provider, "google")
        with patch.dict(os.environ, {"PASA_SEARCH_PROVIDER": "invalid"}):
            with self.assertRaises(ValueError):
                self.agent()
        with self.assertRaises(ValueError):
            self.agent("google")

    def test_url_filtering_deduplication_text_response_and_limits(self):
        urls = ["https://arxiv.org.evil.test/abs/2501.10120", "https://evil.test/arxiv.org/abs/2501.10120",
                "file://arxiv.org/abs/2501.10120", "https://arxiv.org/abs/invalid",
                "https://arxiv.org/abs/2501.10120v2", "https://arxiv.org/pdf/2501.10120.pdf",
                "https://www.arxiv.org/html/2401.12345v1", "https://arxiv.org/abs/2301.12345"]
        server = MCPServer("text", urls)
        with patch.object(paper_search.httpx, "AsyncClient", server.client):
            agent = self.agent()
            self.assertEqual(agent.search_tool.search("query", 2, "bad-date"), [PAPER_ID, "2401.12345"])
            self.assertNotIn("before:", server.calls()[0]["arguments"]["search_queries"][0])
            count = len(server.requests)
            self.assertEqual(agent.search_tool.search("query", 0), [])
            self.assertEqual(len(server.requests), count)

    def test_errors_timeout_and_retry_keep_run_identity_without_fallback(self):
        for mode in ("tool_error", "http_error", "transport_timeout", "malformed", "slow"):
            with self.subTest(mode=mode):
                server = MCPServer(mode)
                with patch.object(paper_search.httpx, "AsyncClient", server.client), \
                        patch.object(self.agent_module, "google_search_arxiv_id") as google:
                    agent = self.agent()
                    if mode == "slow":
                        agent.search_tool.timeout = 0.1
                    with self.assertWarnsRegex(UserWarning, "Parallel search failed"):
                        agent.search_paper(["paper search"])
                    self.assertEqual(agent.papers_queue, [])
                    identity = agent.search_tool.session_id
                    server.mode = "structured"
                    agent.search_tool.timeout = 30
                    agent.search_paper(["retry paper search"])
                    self.assertEqual(agent.root.extra["recall_papers"], ["PaSa"])
                    self.assertEqual({c["arguments"]["session_id"] for c in server.calls()}, {identity})
                    google.assert_not_called()

    def test_concurrent_runs_followups_and_new_run_identity(self):
        server = MCPServer()
        with patch.object(paper_search.httpx, "AsyncClient", server.client):
            first, second = self.agent(), self.agent()
            threads = [threading.Thread(target=a.run) for a in (first, second)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            first.search_paper(["follow up first"])
            second.search_paper(["follow up second"])
        self.assertNotEqual(first.search_tool.session_id, second.search_tool.session_id)
        self.assertNotIn(self.agent().search_tool.session_id, {first.search_tool.session_id, second.search_tool.session_id})
        self.assertEqual(first.root.extra["recall_papers"], ["PaSa"])
        self.assertEqual(second.root.extra["recall_papers"], ["PaSa"])
        for agent in (first, second):
            calls = [c for c in server.calls() if c["arguments"]["session_id"] == agent.search_tool.session_id]
            self.assertEqual(len(calls), 3)
        followups = {c["arguments"]["objective"]: c["arguments"]["session_id"] for c in server.calls()}
        self.assertEqual(followups["Find arXiv papers relevant to: follow up first"], first.search_tool.session_id)
        self.assertEqual(followups["Find arXiv papers relevant to: follow up second"], second.search_tool.session_id)

    def test_cancelled_mcp_call_propagates(self):
        server = MCPServer("slow")
        async def run():
            server.started = asyncio.Event()
            task = asyncio.create_task(self.agent().search_tool._parallel_search("query", 10, None))
            await asyncio.wait_for(server.started.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        with patch.object(paper_search.httpx, "AsyncClient", server.client):
            asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
