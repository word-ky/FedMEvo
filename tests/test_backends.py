import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from fedmevo.federation import Experience, Task, run_federation
from fedmevo.llm import APIBackend, DemoBackend

def test_api_transport_and_federation_against_local_fixture(monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            assert self.headers["Authorization"] == "Bearer test-only-placeholder"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            content = DemoBackend().generate(body["messages"][0]["content"], body["messages"][1]["content"])
            response = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    monkeypatch.setenv("FEDMEVO_TEST_KEY", "test-only-placeholder")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = APIBackend(f"http://127.0.0.1:{server.server_port}/v1", "test-model", "FEDMEVO_TEST_KEY")
        experiences = {"a": [Experience(Task("a1", "a", "radio", "A or B?"), "A")],
                       "b": [Experience(Task("b1", "b", "airflow", "A or B?"), "B")]}
        result = run_federation(backend, experiences, {},
                                [Task("ta", "a", "airflow", "A or B?"), Task("tb", "b", "radio", "A or B?")])
        assert len(result["predictions"]) == 2
        assert result["communication"]["uplink_bytes"] > 0
        assert requests and all(r["model"] == "test-model" for r in requests)
        assert "test-only-placeholder" not in json.dumps(result)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

def test_missing_api_key_is_not_silently_replaced_by_demo(monkeypatch):
    monkeypatch.delenv("FEDMEVO_MISSING_TEST_KEY", raising=False)
    with pytest.raises(KeyError):
        APIBackend("https://example.invalid/v1", "test-model", "FEDMEVO_MISSING_TEST_KEY")
