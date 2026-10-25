"""Generate the runtime's loader table from the compiler operator records."""

from __future__ import annotations

import argparse
from pathlib import Path

from tigris.operator_contract import ATTRIBUTES, OPERATOR_CONTRACT


FLAGS = (
    "binary", "axis_unary", "axis_binary", "row", "height",
    "spatial", "resample", "whole_operand", "fused", "fused_float",
)


def render() -> str:
    rows = sorted(OPERATOR_CONTRACT.values(), key=lambda row: row.opcode)
    output = [
        "/* Generated from the compiler's operator table. Do not edit. */",
        "#ifndef TIGRIS_OP_CONTRACT_H",
        "#define TIGRIS_OP_CONTRACT_H",
        "",
        "typedef struct {",
        "    uint8_t inputs[3];",
        "    uint8_t output;",
        "} dtype_signature_t;",
        "typedef tigris_error_t (*tigris_op_validator_t)(const tigris_plan_t *,",
        "                                                const tigris_op_t *, uint16_t);",
        "typedef struct {",
        "    dtype_signature_t dtype;",
        "    uint8_t min_inputs;",
        "    uint8_t max_inputs;",
        "    uint8_t min_outputs;",
        "    uint8_t max_outputs;",
        "    uint32_t allowed_attributes;",
        "    uint32_t required_attributes;",
        "    uint16_t flags;",
        "    uint8_t executor;",
        "    uint8_t linked_f32;",
        "    uint8_t linked_s8;",
        "    uint8_t linked_i32;",
        "    tigris_op_validator_t validate;",
        "} tigris_op_contract_t;",
        "typedef struct {",
        "    uint8_t min_len;",
        "    uint8_t max_len;",
        "    uint8_t step;",
        "    uint8_t exact_len;",
        "} tigris_attribute_contract_t;",
        "",
    ]
    output += [f"#define TIGRIS_CONTRACT_{flag.upper()} 0x{1 << index:04x}u"
               for index, flag in enumerate(FLAGS)]
    output += [
        "",
    ]
    output.append("#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L")
    for row in rows:
        output.append(f"_Static_assert((uint32_t){row.c_name} == {row.opcode}u, \"operator opcode\");")
    output += ["#endif", ""]
    for handler in sorted({row.validator for row in rows if row.validator is not None}):
        output += [f"static tigris_error_t {handler}(const tigris_plan_t *plan,",
                   "                                         const tigris_op_t *op, uint16_t op_index);"]
    output += ["", "static const tigris_op_contract_t *tigris_op_contract(uint8_t opcode)",
               "{", f"    static const tigris_op_contract_t rows[(uint32_t){rows[-1].c_name} + 1u] = {{"]
    for row in rows:
        slots = tuple(row.signature.inputs)
        slots += (slots[-1],) * (3 - len(slots))
        flags = " | ".join(f"TIGRIS_CONTRACT_{flag.upper()}" for flag in FLAGS if flag in row.modes) or "0u"
        allowed = sum(1 << kind for kind in row.attributes)
        required = sum(1 << kind for kind in row.required_attributes)
        dtype = ("{{" + f"{slots[0]}u, {slots[1]}u, {slots[2]}u" + "}, "
                 + f"{row.signature.outputs[0]}u" + "}")
        kernel = "1u, 1u, 1u" if row.executor else (
            f"TIGRIS_KERNEL_F32_{row.c_name.removeprefix('TIGRIS_OP_')}, "
            f"TIGRIS_KERNEL_S8_{row.c_name.removeprefix('TIGRIS_OP_')}, "
            "TIGRIS_KERNEL_I32")
        output.append(
            f"        [{row.c_name}] = {{" + dtype + f", {row.inputs[0]}u, {row.inputs[1]}u, "
            f"{row.outputs[0]}u, {row.outputs[1]}u, "
            f"0x{allowed:08x}u, 0x{required:08x}u, {flags}, "
            f"{int(row.executor)}u, {kernel}, {row.validator or 'NULL'}" + "},"
        )
    output += [
        "    };",
        "    const tigris_op_contract_t *result = NULL;",
        "    if (opcode < sizeof(rows) / sizeof(rows[0]))",
        "        result = &rows[opcode];",
        "    return result;",
        "}",
        "",
        "static const tigris_attribute_contract_t *tigris_attribute_contract(uint8_t kind)",
        "{",
        f"    static const tigris_attribute_contract_t rows[{max(ATTRIBUTES) + 1}u] = {{",
    ]
    for code, attr in sorted(ATTRIBUTES.items()):
        exact = attr.lengths[0] if len(attr.lengths) == 1 else 0
        output.append(f"        [{code}] = {{{attr.minimum}u, {attr.maximum}u, "
                      f"{attr.step}u, {exact}u}},")
    output += [
        "    };",
        "    const tigris_attribute_contract_t *result = NULL;",
        "    if (kind < sizeof(rows) / sizeof(rows[0]))",
        "        result = &rows[kind];",
        "    return result;",
        "}",
        "#endif",
        "",
    ]
    return "\n".join(output)


def render_config() -> str:
    rows = sorted(OPERATOR_CONTRACT.values(), key=lambda row: row.opcode)
    output = [
        "/* Which operator kernels the runtime links. A build that defines",
        " * TIGRIS_OP_CONFIG as a quoted header path reads it first; such a header sets",
        " * TIGRIS_KERNEL_DEFAULT to 0 and enables the kernels a plan uses, so the",
        " * linker can drop the rest. Without it every kernel is linked. A plan that",
        " * needs a kernel this build leaves out fails to load with",
        " * TIGRIS_ERR_KERNEL_DISABLED. Generated from the compiler's operator table. */",
        "#ifndef TIGRIS_OP_CONFIG_H",
        "#define TIGRIS_OP_CONFIG_H",
        "",
        "#ifdef TIGRIS_OP_CONFIG",
        "#include TIGRIS_OP_CONFIG",
        "/* A kernel left out stays defined without a caller; the linker drops it. */",
        "#if defined(__GNUC__)",
        "#pragma GCC diagnostic ignored \"-Wunused-function\"",
        "#pragma GCC diagnostic ignored \"-Wunused-const-variable\"",
        "#elif defined(_MSC_VER)",
        "#pragma warning(disable : 4505)",
        "#endif",
        "#endif",
        "",
        "#ifndef TIGRIS_KERNEL_DEFAULT",
        "#define TIGRIS_KERNEL_DEFAULT 1",
        "#endif",
        "",
        "/* int32 arithmetic, comparisons and casts. */",
        "#ifndef TIGRIS_KERNEL_I32",
        "#define TIGRIS_KERNEL_I32 TIGRIS_KERNEL_DEFAULT",
        "#endif",
    ]
    titles = {"F32": "/* float32 kernels. */",
              "S8": "/* int8 kernels, including the accelerated adapters. */"}
    for family in ("F32", "S8"):
        output += ["", titles[family]]
        for row in rows:
            if row.executor:
                continue
            macro = f"TIGRIS_KERNEL_{family}_{row.c_name.removeprefix('TIGRIS_OP_')}"
            output += [f"#ifndef {macro}", f"#define {macro} TIGRIS_KERNEL_DEFAULT", "#endif"]
    return "\n".join(output + ["", "#endif /* TIGRIS_OP_CONFIG_H */", ""])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = {
        args.runtime / "src/tigris_op_contract.h": render(),
        args.runtime / "src/tigris_op_config.h": render_config(),
    }
    if args.check:
        for path, content in generated.items():
            if not path.exists() or path.read_text() != content:
                print(f"stale operator contract: {path}")
                return 1
        print("Operator contract headers are current.")
    else:
        for path, content in generated.items():
            path.write_text(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
