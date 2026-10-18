"""Tests for the JSON plan emitter."""

import json

from tigris import SCHEMA_VERSION
from tigris.emitters.plan_json import emit_plan_json, plan_json_str

from conftest import full_pipeline


def test_plan_json_roundtrip(linear_3op_path, tmp_path):
    ag = full_pipeline(linear_3op_path)
    out = tmp_path / "test.plan.json"
    emit_plan_json(ag, out)
    plan = json.loads(out.read_text())

    assert plan["version"] == SCHEMA_VERSION
    assert plan["model"]["name"] == "linear_3op"
    assert len(plan["ops"]) == 3
    assert len(plan["inputs"]) == 1
    assert plan["inputs"][0]["name"] == "input"
    assert plan["inputs"][0]["dtype"] == "float32"
    assert plan["inputs"][0]["shape"] == [1, 64]
    assert plan["inputs"][0]["size_bytes"] == 1 * 64 * 4


def test_plan_json_has_stages(diamond_path):
    ag = full_pipeline(diamond_path, budget=512, spatial=False)
    plan = json.loads(plan_json_str(ag))

    assert "stages" in plan
    assert len(plan["stages"]) >= 2
    assert plan["memory"]["budget"] == 512
