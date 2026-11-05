#!/usr/bin/env python3
"""Submit an API-format workflow from example_workflows to a running ComfyUI."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def _request(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode())


def submit(server: str, workflow: Path) -> str:
    prompt = json.loads(workflow.read_text(encoding="utf-8"))
    body = _request(server.rstrip("/") + "/prompt", {"prompt": prompt})
    prompt_id = body.get("prompt_id")
    if not prompt_id:
        raise SystemExit(f"Submit failed: {body}")
    print(f"prompt_id={prompt_id}")
    deadline = time.time() + 30 * 60
    while time.time() < deadline:
        history = _request(server.rstrip("/") + f"/history/{prompt_id}")
        if prompt_id in history:
            status = history[prompt_id].get("status", {})
            print(json.dumps(status, ensure_ascii=False, indent=2))
            outputs = history[prompt_id].get("outputs", {})
            print(json.dumps(outputs, ensure_ascii=False, indent=2)[:4000])
            if status.get("status_str") == "error":
                raise SystemExit(1)
            return prompt_id
        time.sleep(2)
    raise SystemExit("Timed out waiting for /history")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    args = parser.parse_args()
    submit(args.server, args.workflow)


if __name__ == "__main__":
    main()
