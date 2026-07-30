#!/usr/bin/env python3
"""Generate or verify the release-carried TiGrIS plan-schema artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tigris import (
    SCHEMA_VERSION,
    TILE_AXIS_HEIGHT_OR_LENGTH,
    TILE_AXIS_NONE,
    TILE_AXIS_WIDTH,
)
from tigris.emitters.binary import defs


def schema_package() -> dict[str, object]:
    """Return the deterministic machine-readable current wire contract."""
    return {
        "artifact_format": 1,
        "endianness": "little",
        "magic_ascii": defs.MAGIC.decode("ascii"),
        "op_attribute_types": {
            "transpose_perm": defs.OP_ATTR_TRANSPOSE_PERM,
        },
        "op_types": dict(sorted(defs.OP_TYPE_MAP.items())),
        "record_sizes": {
            "header": defs.HEADER_SIZE,
            "op": defs.OP_SIZE,
            "op_attribute": defs.OP_ATTRIBUTE_SIZE,
            "quant_param": defs.QUANT_PARAM_SIZE,
            "section_entry": defs.SECTION_ENTRY_SIZE,
            "stage": defs.STAGE_SIZE,
            "tensor": defs.TENSOR_SIZE,
            "tile_plan": defs.TILE_PLAN_SIZE,
            "weight_block": defs.WEIGHT_BLOCK_SIZE,
            "weight_entry": defs.WEIGHT_ENTRY_SIZE,
        },
        "schema_version": SCHEMA_VERSION,
        "section_alignment": defs.PLAN_SECTION_ALIGNMENT,
        "tile_axes": {
            "height_or_length": TILE_AXIS_HEIGHT_OR_LENGTH,
            "none": TILE_AXIS_NONE,
            "width_reserved": TILE_AXIS_WIDTH,
        },
        "section_types": {
            "index_pool": defs.SEC_INDEX_POOL,
            "op_attributes": defs.SEC_OP_ATTRIBUTES,
            "ops": defs.SEC_OPS,
            "quant_params": defs.SEC_QUANT_PARAMS,
            "shape_pool": defs.SEC_SHAPE_POOL,
            "stages": defs.SEC_STAGES,
            "strings": defs.SEC_STRINGS,
            "tensors": defs.SEC_TENSORS,
            "tile_plans": defs.SEC_TILE_PLANS,
            "weight_blocks": defs.SEC_WEIGHT_BLOCKS,
            "weights": defs.SEC_WEIGHTS,
        },
    }


def encoded_package() -> str:
    return json.dumps(schema_package(), indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", type=Path, metavar="PATH")
    group.add_argument("--output", type=Path, metavar="PATH")
    args = parser.parse_args()
    encoded = encoded_package()
    if args.check:
        if not args.check.is_file() or args.check.read_text() != encoded:
            parser.error(f"schema artifact is stale: {args.check}")
        print(f"Schema artifact is current: {args.check}")
    elif args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
