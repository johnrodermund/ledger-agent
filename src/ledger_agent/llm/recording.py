"""Request hashing and on-disk recordings for the model client.

A recording is one JSON file per request at ``<recordings>/<namespace>/<key>.json``.
Files are written deterministically (sorted keys, fixed indent, trailing newline)
so that diffs stay reviewable. A file holds the key, the model, the response and a
little metadata, not the request, which is reproduced by the run and is large: no API
key and no HTTP headers ever reach this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


class RecordingMissing(Exception):
    """Replay mode found no recording for a request."""

    def __init__(self, key: str, namespace: str, path: Path):
        self.key = key
        self.namespace = namespace
        self.path = path
        super().__init__(
            f"No recording for request {key} in namespace '{namespace}' "
            f"(expected {path.as_posix()}). Re-record with LEDGER_LLM_MODE=record "
            f"and a configured Anthropic API key, then commit the new file."
        )


def request_payload(req: Any) -> dict[str, Any]:
    return {
        "model": req.model,
        "system": req.system,
        "messages": req.messages,
        "tools": req.tools,
        "tool_choice": req.tool_choice,
    }


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_key(req: Any) -> str:
    """SHA-256 of the canonical JSON of the request, independent of dict ordering."""
    return hashlib.sha256(canonical_json(request_payload(req)).encode("utf-8")).hexdigest()


def recording_path(root: Path, namespace: str, key: str) -> Path:
    return Path(root) / namespace / f"{key}.json"


def recording_exists(root: Path, namespace: str, req: Any) -> bool:
    return recording_path(root, namespace, request_key(req)).is_file()


def _metadata(namespace: str, req: Any) -> dict[str, Any]:
    """Enough to tell recordings apart when debugging. No clock, so a re-record of the
    same answer gives the same file."""
    return {
        "namespace": namespace,
        "message_count": len(req.messages),
        "tools": [tool["name"] for tool in req.tools],
        "tool_choice": req.tool_choice,
    }


def save_recording(root: Path, namespace: str, req: Any, response: Any) -> Path:
    key = request_key(req)
    path = recording_path(root, namespace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"key": key, "model": req.model, "response": asdict(response),
           "metadata": _metadata(namespace, req)}
    text = json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    # Through a temporary name: an interrupted run must not leave a file that looks whole.
    partial = path.with_suffix(".partial")
    partial.write_text(text, encoding="utf-8", newline="\n")
    partial.replace(path)
    return path


def load_recording(root: Path, namespace: str, req: Any) -> dict[str, Any]:
    key = request_key(req)
    path = recording_path(root, namespace, key)
    if not path.is_file():
        raise RecordingMissing(key, namespace, path)
    return json.loads(path.read_text(encoding="utf-8"))["response"]
