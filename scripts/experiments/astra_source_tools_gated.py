"""Opt-in completion tool contract; legacy source tools remain byte-for-byte intact."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Private module instances work inside the sealed bundle without a repo import.
legacy = load_sibling("astra_source_tools")
gate = load_sibling("completion_gate")
SERVER_INSTRUCTIONS = legacy.SERVER_INSTRUCTIONS + (
    " Completion policy completion_gate_v1: a final claim is accepted only when notation, "
    "meter and executable-route issues are resolved. Save feedback contains completion_gate "
    "with exact issue IDs and the ABC hash. Inspect nearby notation and parallel passages "
    "in this PDF, correct the draft and save again. Never pad rests mechanically. "
    "If the manuscript itself has an irregular bar or unpaired pickup, record a source "
    "exception in resolutions citing an earlier inspect_source call. Include the current "
    "abc_sha256, issue_id, category, location, adopted_choice, reason and uncertainty. "
    "An uncertain pitch is a documented best-effort choice, not permission to ignore meter. "
    "Preserve partial progress if time runs out; do not claim human approval."
)
TOOLS = copy.deepcopy(legacy.TOOLS)
TOOLS[1]["inputSchema"]["properties"]["resolutions"] = {
    "type": "array",
    "maxItems": 128,
    "items": {
        "type": "object",
        "properties": {
            **{
                key: {"type": "string", "minLength": 1, "maxLength": 4000}
                for key in gate.RESOLUTION_FIELDS
                if key != "inspection_call"
            },
            "category": {"type": "string", "enum": list(gate.EXCEPTION_CATEGORIES)},
            "inspection_call": {"type": "integer", "minimum": 1},
        },
        "required": list(gate.RESOLUTION_FIELDS),
        "additionalProperties": False,
    },
}
TOOLS[1]["description"] += (
    " completion_gate determines readiness. Resolve every issue or record an evidenced "
    "source-meter exception in resolutions; unknown or stale exceptions block completion."
)


class SourceTools(legacy.SourceTools):
    def save(self, args):
        ordinary = {key: value for key, value in args.items() if key != "resolutions"}
        resolutions = args.get("resolutions", [])
        if len(json.dumps(resolutions).encode()) > legacy.MAX_DRAFT_BYTES:
            raise ValueError("Resolution data exceeds the checkpoint allowance")
        _, evidence = super().save(ordinary)
        directory = self.artifacts / evidence["checkpoint"]
        legacy.write_json(directory / "completion_resolutions.json", resolutions)
        result = gate.evaluate_checkpoint(directory, self.artifacts)
        legacy.write_json(directory / "completion_gate.json", result)
        feedback = json.loads((directory / "validation.json").read_text())
        feedback["completion_gate"] = result
        legacy.write_json(directory / "validation.json", feedback)
        evidence = {
            **evidence,
            "completion_policy": gate.POLICY_ID,
            "completion_ready": result["completion_ready"],
        }
        return [{"type": "text", "text": json.dumps(feedback)}], evidence


def serve(root):
    # Only this privately loaded instance is changed. The legacy module imported
    # by the ordinary runner, its file, schema and instructions are untouched.
    legacy.SourceTools = SourceTools
    legacy.TOOLS = TOOLS
    legacy.SERVER_INSTRUCTIONS = SERVER_INSTRUCTIONS
    legacy.serve(root)


if __name__ == "__main__":
    serve(Path(sys.argv[1]))
