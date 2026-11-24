#!/usr/bin/env python3
"""Validate the declarative gfx1030 semantic registry and generate its Rust tables.

  validate   check crates/isa/semantics/rdna2.toml against its closed vocabulary
             and the pinned AMD XML snapshot (crates/isa/data/rdna2.jsonl)
  generate   validate, then write crates/isa/src/semantics_gen.rs
  --check    validate and regenerate in memory; exit 1 if the committed file
             differs. Never writes.

The registry is data only. Behaviour is chosen from a closed set of primitives
(crates/isa/src/primitive.rs) by the declared operation, signedness and SCC
effect, never from a mnemonic. Only the Python standard library is used
(tomllib needs Python 3.11 or newer). See docs/shaders/semantic-registry.md.
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - depends on the interpreter
    sys.exit("generate_semantics.py needs Python 3.11 or newer (tomllib)")

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "crates/isa/semantics/rdna2.toml"
SNAPSHOT = ROOT / "crates/isa/data/rdna2.jsonl"
RUST = ROOT / "crates/isa/src/semantics_gen.rs"

SCHEMA = 1
ARCHITECTURE = "gfx1030"

FORM_KEYS = {
    "name", "family", "opcode", "variant", "operation", "signedness", "carry_in",
    "width", "scc", "vcc", "exec", "wave", "operands", "spec", "verify",
}
TOP_KEYS = {"schema", "architecture", "source_sha256", "documents", "operand_rules", "form"}

OPERATIONS = {
    "add": "Add", "sub": "Sub", "and": "And", "or": "Or", "xor": "Xor",
    "andn2": "AndNot", "orn2": "OrNot", "nand": "Nand", "nor": "Nor", "xnor": "Xnor",
    "lshl": "ShiftLeft", "lshr": "LogicalShiftRight", "ashr": "ArithmeticShiftRight",
    "compare": "Compare",
    "not": "Not", "brev": "ReverseBits", "bcnt0": "CountZeros", "bcnt1": "CountOnes",
    "ff0": "FindFirstZero", "ff1": "FindFirstOne", "flbit": "FindLastBit",
    "sext": "SignExtend", "abs": "Absolute",
    "min": "Minimum", "max": "Maximum", "cselect": "Select", "bfm": "BitfieldMask",
    "mul": "Multiply", "mul_hi": "MultiplyHigh", "bfe": "BitfieldExtract",
    "absdiff": "AbsoluteDifference", "lshl_add": "ShiftLeftAdd",
    "mul24": "Multiply24", "mul_hi24": "MultiplyHigh24",
}
# Unary operations read one source; they need an operand rule of the unary kind.
UNARY = {"not", "brev", "bcnt0", "bcnt1", "ff0", "ff1", "flbit", "sext", "abs"}
# Sign extension reads only the low source_bits of S0 (required for sext only).
SOURCE_BITS_KEY = "source_bits"
SOURCE_BITS = {8, 16}
# Compares take a relation (S0 on the left) and set SCC to whether it holds.
RELATION_KEY = "relation"
RELATIONS = {
    "eq": "Equal", "lg": "NotEqual", "gt": "Greater", "ge": "GreaterOrEqual",
    "lt": "Less", "le": "LessOrEqual",
}
# (relation, signedness) -> primitive. Equality does not depend on signedness,
# so both declared signednesses of EQ and LG share one primitive.
COMPARE_PRIMITIVES = {
    ("eq", "signed"): "eq_b32", ("eq", "unsigned"): "eq_b32",
    ("lg", "signed"): "ne_b32", ("lg", "unsigned"): "ne_b32",
    ("gt", "signed"): "gt_i32", ("gt", "unsigned"): "gt_u32",
    ("ge", "signed"): "ge_i32", ("ge", "unsigned"): "ge_u32",
    ("lt", "signed"): "lt_i32", ("lt", "unsigned"): "lt_u32",
    ("le", "signed"): "le_i32", ("le", "unsigned"): "le_u32",
}
# Logical operations take untyped bits; arithmetic ones never do. Shifts take
# the signedness their primitive names, and their S1 is an unsigned count.
LOGICAL = {
    "and", "or", "xor", "andn2", "orn2", "nand", "nor", "xnor",
    "not", "brev", "bcnt0", "bcnt1", "ff0", "ff1", "cselect", "bfm",
}
ARITHMETIC = {
    "add", "sub", "min", "max", "mul", "mul_hi", "bfe", "absdiff", "lshl_add", "mul24", "mul_hi24",
}
SHIFTS = {"lshl", "lshr", "ashr"}
# Operations whose S1 is an unsigned control word (a shift count or a bit-field
# descriptor), so the XML gives it FMT_NUM_U32 whatever the form's signedness.
CONTROL_S1 = SHIFTS | {"bfe"}
# Operations that always read SCC: the select condition. They need carry_in scc.
SCC_READERS = {"cselect"}
# XML number formats a source may have for each declared signedness.
SOURCE_FORMATS = {
    "signed": {"FMT_NUM_I32"},
    "unsigned": {"FMT_NUM_U32"},
    "bitwise": {"FMT_NUM_U32", "FMT_NUM_B32"},
}
# Optional keys: required exactly when the condition holds.
SHIFT_KEY = "shift_count_bits"
# Shift-and-add declares its fixed S0 shift (required for lshl_add only).
SHIFT_AMOUNT_KEY = "shift_amount"
SHIFT_AMOUNTS = {1, 2, 3, 4}
# Vector forms whose guide pseudocode reads S1 before S0 (D = op(S1, S0))
# declare it; the XML records no operand order.
SOURCE_ORDER_KEY = "source_order"
SOURCE_ORDERS = {"reversed"}
SIGNEDNESS = {"unsigned": "Unsigned", "signed": "Signed", "bitwise": "Bitwise"}
CARRY_IN = {"none": "None", "scc": "Scc"}
SCC = {
    "unchanged": "Unchanged",
    "carry_out": "CarryOut",
    "borrow": "Borrow",
    "signed_overflow": "SignedOverflow",
    "nonzero": "Nonzero",
    "relation": "Relation",
    "selected_src0": "SelectedSrc0",
}
MASK = {"unchanged": "Unchanged"}
WAVE = {"any": "Any", "wave32": "Wave32"}
# The decoder has no wave-size input at the registry dispatch yet, so a Wave32
# restriction could be declared but not enforced. Reject it until it can be.
ENFORCED_WAVES = {"any"}
EVIDENCE = {
    "reference": "Reference",
    "independent": "Independent",
    "metal": "Metal",
    "llvm_encoding": "LlvmEncoding",
}
DOCUMENTS = {"guide": "Guide", "xml": "Xml"}
# The public documents spec references may cite, pinned so a citation cannot
# silently point at another file.
PINNED_DOCUMENTS = {
    "guide": {
        "title": "AMD RDNA 2 Instruction Set Architecture Reference Guide",
        "file": "rdna2-shader-instruction-set-architecture.pdf",
        "pages": 291,
    },
    "xml": {
        "title": "AMD RDNA 2 machine-readable ISA (pinned XML)",
        "file": "amdgpu_isa_rdna2.xml",
    },
}
WIDTHS = {32}

# (operation, signedness) -> (required SCC effect, primitive, takes carry-in).
# A form whose declaration is not a row here has no primitive and is rejected.
PRIMITIVES = {
    ("add", "unsigned"): ("carry_out", "add_carry_u32", True),
    ("sub", "unsigned"): ("borrow", "sub_borrow_u32", True),
    ("add", "signed"): ("signed_overflow", "add_overflow_i32", False),
    ("sub", "signed"): ("signed_overflow", "sub_overflow_i32", False),
    ("and", "bitwise"): ("nonzero", "and_nonzero_b32", False),
    ("or", "bitwise"): ("nonzero", "or_nonzero_b32", False),
    ("xor", "bitwise"): ("nonzero", "xor_nonzero_b32", False),
    ("andn2", "bitwise"): ("nonzero", "andn2_nonzero_b32", False),
    ("orn2", "bitwise"): ("nonzero", "orn2_nonzero_b32", False),
    ("nand", "bitwise"): ("nonzero", "nand_nonzero_b32", False),
    ("nor", "bitwise"): ("nonzero", "nor_nonzero_b32", False),
    ("xnor", "bitwise"): ("nonzero", "xnor_nonzero_b32", False),
    ("lshl", "bitwise"): ("nonzero", "lshl_nonzero_b32", False),
    ("lshr", "unsigned"): ("nonzero", "lshr_nonzero_b32", False),
    ("ashr", "signed"): ("nonzero", "ashr_nonzero_i32", False),
    ("not", "bitwise"): ("nonzero", "not_nonzero_b32", False),
    ("brev", "bitwise"): ("unchanged", "brev_b32", False),
    ("bcnt0", "bitwise"): ("nonzero", "bcnt0_nonzero_b32", False),
    ("bcnt1", "bitwise"): ("nonzero", "bcnt1_nonzero_b32", False),
    ("ff0", "bitwise"): ("unchanged", "ff0_b32", False),
    ("ff1", "bitwise"): ("unchanged", "ff1_b32", False),
    ("flbit", "bitwise"): ("unchanged", "flbit_b32", False),
    ("flbit", "signed"): ("unchanged", "flbit_i32", False),
    ("sext", "signed"): ("unchanged", "sext_i32", False),
    ("abs", "signed"): ("nonzero", "abs_nonzero_i32", False),
    ("min", "signed"): ("selected_src0", "min_i32", False),
    ("min", "unsigned"): ("selected_src0", "min_u32", False),
    ("max", "signed"): ("selected_src0", "max_i32", False),
    ("max", "unsigned"): ("selected_src0", "max_u32", False),
    ("cselect", "bitwise"): ("unchanged", "cselect_b32", True),
    ("bfm", "bitwise"): ("unchanged", "bfm_b32", False),
    ("mul", "signed"): ("unchanged", "mul_lo_i32", False),
    ("mul_hi", "unsigned"): ("unchanged", "mul_hi_u32", False),
    ("mul_hi", "signed"): ("unchanged", "mul_hi_i32", False),
    ("bfe", "unsigned"): ("nonzero", "bfe_nonzero_u32", False),
    ("bfe", "signed"): ("nonzero", "bfe_nonzero_i32", False),
    ("absdiff", "signed"): ("nonzero", "absdiff_nonzero_i32", False),
    ("lshl_add", "unsigned"): ("carry_out", "lshl_add_carry_u32", False),
}

# Vector (VOP2) forms: (operation, signedness) -> the per-lane VectorBinary their
# MSL lowering uses. They write no SCC, VCC or EXEC, so scc must be unchanged.
VECTOR_PRIMITIVES = {
    ("add", "unsigned"): "Add",
    ("sub", "unsigned"): "Sub",
    ("and", "bitwise"): "And",
    ("or", "bitwise"): "Or",
    ("xor", "bitwise"): "Xor",
    ("xnor", "bitwise"): "Xnor",
    ("lshl", "bitwise"): "ShiftLeft",
    ("lshr", "unsigned"): "LogicalShiftRight",
    ("ashr", "signed"): "ArithmeticShiftRight",
    ("min", "signed"): "MinSigned",
    ("min", "unsigned"): "MinUnsigned",
    ("max", "signed"): "MaxSigned",
    ("max", "unsigned"): "MaxUnsigned",
    ("mul24", "signed"): "Mul24Signed",
    ("mul24", "unsigned"): "Mul24Unsigned",
    ("mul_hi24", "signed"): "MulHi24Signed",
    ("mul_hi24", "unsigned"): "MulHi24Unsigned",
}

# Operand rules the decoder implements, each exactly once. A registry rule must
# match one of these shapes; anything else is an unimplemented rule.
UNSUPPORTED_COMBOS = {
    "dst_vcc", "dst_exec", "dst_ttmp", "dst_m0", "dst_null", "src_vcc", "src_exec",
    "src_m0", "src_ttmp", "src_float_inline", "src_special", "two_literals",
}
# Every combo name a rule may list (VOP2 also names vcc_hi and NULL sources,
# which SOP2 groups under src_vcc and src_special).
ALL_COMBOS = UNSUPPORTED_COMBOS | {"src_vcc_hi", "src_null"}
OPERAND_RULES = {
    "sop2_sgpr": {
        "rust": "Sop2Sgpr",
        "enum": "ScalarAluOp",
        "shape": {
            "family": "SOP2",
            "dst": {"field": "SDST", "class": "sgpr", "min": 0, "max": 105},
            "src": [
                {"field": "SSRC0", "class": "scalar_source"},
                {"field": "SSRC1", "class": "scalar_source"},
            ],
            "max_literals": 1,
        },
        # Every one of these is rejected by the decoder's SOP2 operand checks.
        "unsupported": UNSUPPORTED_COMBOS,
        "kind": "alu",
    },
    "sopc_scalar": {
        "rust": "SopcScalar",
        "enum": "ScalarCompareOp",
        "shape": {
            "family": "SOPC",
            "src": [
                {"field": "SSRC0", "class": "scalar_source"},
                {"field": "SSRC1", "class": "scalar_source"},
            ],
            "max_literals": 1,
        },
        # SOPC has no destination field; the decoder rejects these sources.
        "unsupported": {combo for combo in UNSUPPORTED_COMBOS if not combo.startswith("dst_")},
        "kind": "compare",
    },
    "sop1_sgpr": {
        "rust": "Sop1Sgpr",
        "enum": "ScalarUnaryOp",
        "shape": {
            "family": "SOP1",
            "dst": {"field": "SDST", "class": "sgpr", "min": 0, "max": 105},
            "src": [{"field": "SSRC0", "class": "scalar_source"}],
            "max_literals": 1,
        },
        # One source, so two literals cannot occur; the rest are rejected.
        "unsupported": {combo for combo in UNSUPPORTED_COMBOS if combo != "two_literals"},
        "kind": "unary",
    },
    "vop2_vgpr": {
        "rust": "Vop2Vgpr",
        "enum": "VectorAluOp",
        "shape": {
            "family": "VOP2",
            "dst": {"field": "VDST", "class": "vgpr", "min": 0, "max": 255},
            "src": [
                {"field": "SRC0", "class": "vector_source"},
                {"field": "VSRC1", "class": "vgpr"},
            ],
            "max_literals": 1,
        },
        # vector_source = VGPR, s0-s105, vcc_lo, inline integer -16..64 or a
        # literal; the decoder rejects these SRC0 selectors.
        "unsupported": {
            "src_vcc_hi", "src_exec", "src_m0", "src_ttmp", "src_null", "src_float_inline",
            "src_special",
        },
        "kind": "vector",
    },
}

VARIANT = re.compile(r"^[A-Z][A-Za-z0-9]*$")
RESERVED_VARIANTS = {"Self", "ALL"}
U16_MAX = 0xFFFF
NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
SECTION = re.compile(r"^[0-9]+(\.[0-9]+)*$")


def load_registry(path=REGISTRY):
    """Parse the TOML registry into a dict."""
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def load_snapshot(path=SNAPSHOT):
    """Index the normalised XML snapshot: source line, encodings, default variants."""
    source = None
    encodings = {}
    variants = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            kind = record.get("kind")
            if kind == "source":
                source = record
            elif kind == "encoding":
                encodings[record["name"]] = record
            elif kind == "variant" and record.get("condition") == "default":
                variants[(record["instruction"], record["encoding"])] = record
    return {"source": source or {}, "encodings": encodings, "variants": variants}


def _family_encoding(family):
    return "ENC_" + family


def _rust_family(family):
    return family[0] + family[1:].lower()


def _check_rule(name, rule, errors):
    implemented = OPERAND_RULES.get(name)
    if implemented is None:
        errors.append(f"operand rule {name}: not implemented by the decoder")
        return
    unsupported = rule.get("unsupported", [])
    unknown = sorted(set(unsupported) - ALL_COMBOS)
    if unknown:
        errors.append(f"operand rule {name}: unknown unsupported combos {unknown}")
    if len(set(unsupported)) != len(unsupported):
        errors.append(f"operand rule {name}: duplicate unsupported combos")
    if set(unsupported) != implemented["unsupported"]:
        errors.append(
            f"operand rule {name}: unsupported combos differ from those the decoder rejects"
        )
    declared = {key: value for key, value in rule.items() if key != "unsupported"}
    if declared != implemented["shape"]:
        errors.append(
            f"operand rule {name}: declaration differs from the decoder's implemented rule"
        )


def _check_xml(form, rule, snapshot, errors):
    label = form["name"]
    encoding_name = _family_encoding(form["family"])
    encoding = snapshot["encodings"].get(encoding_name)
    if encoding is None:
        errors.append(f"{label}: family {form['family']} has no XML encoding {encoding_name}")
        return
    variant = snapshot["variants"].get((form["name"], encoding_name))
    if variant is None:
        errors.append(f"{label}: no default {encoding_name} variant in the XML snapshot")
        return
    if variant["opcode"] != form["opcode"]:
        errors.append(
            f"{label}: opcode {form['opcode']} disagrees with the XML opcode {variant['opcode']}"
        )
    if _identifier(encoding, form["opcode"]) not in encoding["identifiers"]:
        errors.append(f"{label}: opcode {form['opcode']} is not an identifier of {encoding_name}")
    operands = {}
    scc_in = scc_out = False
    implicit = set()
    for field, data_format, kind, size, is_input, is_output, _implicit, _encoded in variant["operands"]:
        if kind == "OPR_SSRC_SPECIAL_SCC":
            scc_in |= is_input
            scc_out |= is_output
        elif field:
            operands[field] = (data_format, size, is_input, is_output)
        else:
            implicit.add(kind)
    # The vocabulary can state only SCC effects (vcc and exec are "unchanged"),
    # so any other implicit XML operand, such as VCC, is undeclared.
    if implicit:
        errors.append(f"{label}: XML implicit operands {sorted(implicit)} are not declared")
    if scc_in != (form["carry_in"] == "scc"):
        errors.append(f"{label}: carry_in {form['carry_in']} contradicts the XML SCC input")
    if scc_out != (form["scc"] != "unchanged"):
        errors.append(f"{label}: scc {form['scc']} contradicts the XML SCC output")
    if rule is None:
        return
    if not isinstance(rule.get("src"), list) or not all(
        isinstance(src, dict) and isinstance(src.get("field"), str) for src in rule["src"]
    ) or ("dst" in rule and not (isinstance(rule["dst"], dict) and "field" in rule["dst"])):
        return  # the rule check already reported the malformed shape
    fields = [(rule["dst"]["field"], False)] if "dst" in rule else []
    fields += [(src["field"], True) for src in rule["src"]]
    shift = form["operation"] in CONTROL_S1 and len(rule["src"]) > 1
    # The control word is the operation's second operand: S1, or S0 for a
    # reversed vector form (D = op(S1, S0), as in v_lshlrev_b32).
    count_index = 0 if SOURCE_ORDER_KEY in form else 1
    count_field = rule["src"][count_index]["field"] if shift else None
    for field, is_source in fields:
        if field not in operands:
            errors.append(f"{label}: XML has no operand {field}")
            continue
        data_format, size, is_input, is_output = operands[field]
        bits = form.get(SOURCE_BITS_KEY) if is_source else None
        expected = bits if type(bits) is int else form["width"]
        if size != expected:
            errors.append(f"{label}: {field} is {size} bits in the XML, declared {expected}")
        if (is_input, is_output) != (is_source, not is_source):
            errors.append(f"{label}: {field} direction disagrees with the XML")
        if field == count_field:
            if data_format != "FMT_NUM_U32":
                errors.append(
                    f"{label}: control operand {field} has XML format {data_format}, not FMT_NUM_U32"
                )
        elif type(bits) is int:
            if data_format != f"FMT_NUM_I{bits}":
                errors.append(f"{label}: {field} has XML format {data_format}, not FMT_NUM_I{bits}")
        elif is_source and data_format not in SOURCE_FORMATS.get(form["signedness"], ()):
            errors.append(
                f"{label}: signedness {form['signedness']} contradicts XML format {data_format}"
            )
    extra = sorted(set(operands) - {field for field, _ in fields})
    if extra:
        errors.append(f"{label}: XML operands {extra} are not covered by rule {form['operands']}")


def _identifier(encoding, opcode):
    fields = dict((name, ranges) for name, ranges in encoding["fields"])
    (offset, size), = fields["OP"]
    base = encoding["identifiers"][0] & ~(((1 << size) - 1) << offset)
    return base | (opcode << offset)


def validate(registry, snapshot):
    """Return every error in the registry; an empty list means it is valid."""
    errors = []
    unknown = sorted(set(registry) - TOP_KEYS)
    if unknown:
        errors.append(f"unknown top-level keys {unknown}")
    if type(registry.get("schema")) is not int or registry["schema"] != SCHEMA:
        errors.append(f"schema {registry.get('schema')!r} is not {SCHEMA}")
    if registry.get("architecture") != ARCHITECTURE:
        errors.append(f"architecture {registry.get('architecture')!r} is not {ARCHITECTURE}")
    if registry.get("source_sha256") != snapshot["source"].get("sha256"):
        errors.append("source_sha256 does not match the XML snapshot")
    documents = registry.get("documents", {})
    for name, document in documents.items():
        if name not in DOCUMENTS:
            errors.append(f"unknown document {name}")
        elif document != PINNED_DOCUMENTS[name]:
            errors.append(f"document {name}: differs from the pinned public document")
    if "guide" not in documents:
        errors.append("the guide document is not declared")
    rules = registry.get("operand_rules", {})
    for name, rule in rules.items():
        _check_rule(name, rule, errors)
    forms = registry.get("form", [])
    if not forms:
        errors.append("the registry declares no forms")
    seen = {"name": set(), "opcode": set(), "variant": set()}
    for index, form in enumerate(forms):
        label = form.get("name", f"form #{index}")
        missing = sorted(FORM_KEYS - set(form))
        extra = sorted(
            set(form) - FORM_KEYS
            - {SHIFT_KEY, SHIFT_AMOUNT_KEY, RELATION_KEY, SOURCE_BITS_KEY, SOURCE_ORDER_KEY}
        )
        if missing:
            errors.append(f"{label}: incomplete, missing {missing}")
        if extra:
            errors.append(f"{label}: unknown keys {extra}")
        if missing:
            continue
        for key, vocabulary in (
            ("operation", OPERATIONS), ("signedness", SIGNEDNESS), ("carry_in", CARRY_IN),
            ("scc", SCC), ("vcc", MASK), ("exec", MASK), ("wave", WAVE),
        ):
            if form[key] not in vocabulary:
                errors.append(f"{label}: unknown {key} {form[key]!r}")
        if not isinstance(form["name"], str) or not NAME.match(form["name"]):
            errors.append(f"{label}: invalid name")
        if not isinstance(form["variant"], str) or not VARIANT.match(form["variant"]):
            errors.append(f"{label}: invalid variant {form['variant']!r}")
        if form["variant"] in RESERVED_VARIANTS:
            errors.append(f"{label}: variant {form['variant']!r} is reserved")
        if type(form["opcode"]) is not int or not 0 <= form["opcode"] <= U16_MAX:
            errors.append(f"{label}: invalid opcode")
        if form["wave"] in WAVE and form["wave"] not in ENFORCED_WAVES:
            errors.append(f"{label}: wave {form['wave']} has no decoder enforcement yet")
        if type(form["width"]) is not int or form["width"] not in WIDTHS:
            errors.append(f"{label}: width {form['width']} is not supported")
        for key, value in (
            ("name", form["name"]),
            ("opcode", (form["family"], form["opcode"])),
            # Variants are names in their operand rule's enum.
            ("variant", (_rule_enum(form), form["variant"])),
        ):
            if value in seen[key]:
                shown = form["variant"] if key == "variant" else value
                errors.append(f"{label}: duplicate {key} {shown}")
            seen[key].add(value)
        if (form["operation"] in LOGICAL and form["signedness"] != "bitwise") or (
            form["operation"] in ARITHMETIC and form["signedness"] == "bitwise"
        ):
            errors.append(
                f"{label}: signedness {form['signedness']} contradicts operation {form['operation']}"
            )
        if form["operation"] in SHIFTS:
            bits = form.get(SHIFT_KEY)
            if type(bits) is not int or not 0 < bits < 32 or (1 << bits) != form["width"]:
                errors.append(
                    f"{label}: {SHIFT_KEY} must be log2 of the width ({form['width']}), got {bits!r}"
                )
        elif SHIFT_KEY in form:
            errors.append(f"{label}: {SHIFT_KEY} on a non-shift operation")
        if form["operation"] == "lshl_add":
            amount = form.get(SHIFT_AMOUNT_KEY)
            if type(amount) is not int or amount not in SHIFT_AMOUNTS:
                errors.append(
                    f"{label}: {SHIFT_AMOUNT_KEY} must be one of {sorted(SHIFT_AMOUNTS)}, got {amount!r}"
                )
        elif SHIFT_AMOUNT_KEY in form:
            errors.append(f"{label}: {SHIFT_AMOUNT_KEY} on a non-lshl_add operation")
        if SOURCE_ORDER_KEY in form:
            order = form[SOURCE_ORDER_KEY]
            if not isinstance(order, str) or order not in SOURCE_ORDERS:
                errors.append(f"{label}: unknown {SOURCE_ORDER_KEY} {order!r}")
            elif _rule_kind(form) != "vector":
                errors.append(f"{label}: {SOURCE_ORDER_KEY} on a non-vector form")
        if form["operation"] in SCC_READERS and form["carry_in"] != "scc":
            errors.append(f"{label}: operation {form['operation']} reads SCC (requires carry_in scc)")
        if form["operation"] == "sext":
            bits = form.get(SOURCE_BITS_KEY)
            if type(bits) is not int or bits not in SOURCE_BITS:
                errors.append(f"{label}: {SOURCE_BITS_KEY} must be one of {sorted(SOURCE_BITS)}, got {bits!r}")
        elif SOURCE_BITS_KEY in form:
            errors.append(f"{label}: {SOURCE_BITS_KEY} on a non-sext operation")
        if form["operation"] == "compare":
            relation = form.get(RELATION_KEY)
            if not isinstance(relation, str) or relation not in RELATIONS:
                errors.append(f"{label}: unknown or missing {RELATION_KEY} {relation!r}")
            elif (relation, form["signedness"]) not in COMPARE_PRIMITIVES:
                errors.append(f"{label}: no primitive implements {relation}/{form['signedness']}")
            if form["scc"] != "relation":
                errors.append(f"{label}: scc {form['scc']} contradicts compare (requires relation)")
            if form["carry_in"] != "none":
                errors.append(f"{label}: carry_in {form['carry_in']} contradicts compare")
            primitive = ("relation", None, False)
        elif _rule_kind(form) == "vector":
            if RELATION_KEY in form:
                errors.append(f"{label}: {RELATION_KEY} on a non-compare operation")
            lowering = VECTOR_PRIMITIVES.get((form["operation"], form["signedness"]))
            primitive = None if lowering is None else ("unchanged", lowering, False)
        else:
            if RELATION_KEY in form:
                errors.append(f"{label}: {RELATION_KEY} on a non-compare operation")
            primitive = PRIMITIVES.get((form["operation"], form["signedness"]))
        if primitive is None:
            errors.append(f"{label}: no primitive implements {form['operation']}/{form['signedness']}")
        else:
            scc, _, takes_carry = primitive
            if form["scc"] != scc:
                errors.append(
                    f"{label}: scc {form['scc']} contradicts {form['operation']}/"
                    f"{form['signedness']} (requires {scc})"
                )
            if form["carry_in"] == "scc" and not takes_carry:
                errors.append(f"{label}: carry_in scc contradicts {form['operation']}/{form['signedness']}")
        rule = rules.get(form["operands"])
        if rule is None:
            errors.append(f"{label}: unknown operand rule {form['operands']!r}")
        elif rule.get("family") != form["family"]:
            errors.append(f"{label}: operand rule {form['operands']} is for {rule.get('family')}")
        elif form["operands"] in OPERAND_RULES and OPERAND_RULES[form["operands"]]["kind"] != (
            "compare" if form["operation"] == "compare"
            else "unary" if form["operation"] in UNARY
            else "vector" if _rule_kind(form) == "vector" and form["operation"] not in UNARY
            else "alu"
        ):
            errors.append(f"{label}: operation {form['operation']} does not fit rule {form['operands']}")
        _check_spec(label, form["spec"], documents, errors)
        verify = form["verify"]
        if not verify:
            errors.append(f"{label}: no verification requirement")
        if any(item not in EVIDENCE for item in verify):
            errors.append(f"{label}: unknown verification requirement in {verify}")
        if len(set(verify)) != len(verify):
            errors.append(f"{label}: duplicate verification requirement")
        _check_xml(form, rule, snapshot, errors)
    return errors


def _implemented_rule(form):
    operands = form.get("operands")
    return OPERAND_RULES.get(operands) if isinstance(operands, str) else None


def _rule_kind(form):
    """The implemented kind of the form's operand rule, or None if unknown."""
    rule = _implemented_rule(form)
    return rule["kind"] if rule else None


def _rule_enum(form):
    """The generated enum of the form's operand rule, or None if unknown."""
    rule = _implemented_rule(form)
    return rule["enum"] if rule else None


def _check_spec(label, spec, documents, errors):
    has_guide = False
    for ref in spec:
        if set(ref) != {"document", "section", "page"}:
            errors.append(f"{label}: spec reference needs exactly document, section, page")
            continue
        if ref["document"] not in documents:
            errors.append(f"{label}: spec cites undeclared document {ref['document']!r}")
        if not SECTION.match(str(ref["section"])):
            errors.append(f"{label}: invalid spec section {ref['section']!r}")
        if type(ref["page"]) is not int or not 0 <= ref["page"] <= U16_MAX:
            errors.append(f"{label}: invalid spec page")
        if ref["document"] == "guide" and type(ref["page"]) is int and ref["page"] > 0:
            has_guide = True
    if not has_guide:
        errors.append(f"{label}: missing a guide section and page reference")


def render(registry, snapshot):
    """Return the complete generated Rust source. Requires a valid registry."""
    forms = registry["form"]
    rule_names = sorted({form["operands"] for form in forms})
    positions = {form["name"]: position for position, form in enumerate(forms)}
    out = [
        "// @generated by crates/isa/generate_semantics.py from crates/isa/semantics/rdna2.toml.",
        "// Do not edit. Regenerate: python3 crates/isa/generate_semantics.py generate",
        "",
        "use crate::meta::EncodingFamily;",
        "use crate::primitive;",
        "use crate::semantics::{",
        "    CarryIn, Document, Evidence, FormSemantics, MaskEffect, Operation, OperandRule, SccEffect,",
        "    Signedness, SpecRef, VectorBinary, Wave,",
        "};",
        "use crate::CompareRelation;",
        "",
        "/// Registry schema of the generated tables.",
        f"pub const GENERATED_SCHEMA: u32 = {registry['schema']};",
    ]
    for rule_name in rule_names:
        selected = [form for form in forms if form["operands"] == rule_name]
        out += _render_enum(rule_name, selected, snapshot, positions)
    out += [
        "",
        "/// Identity of every registry form, in registry order.",
        f"pub const FORM_IDS: [(&str, EncodingFamily); {len(forms)}] = [",
    ]
    for form in forms:
        out.append(f"    (\"{form['name']}\", EncodingFamily::{_rust_family(form['family'])}),")
    out += [
        "];",
        "",
        "/// One validated row per registry form, in registry order.",
        f"pub static FORMS: [FormSemantics; {len(forms)}] = [",
    ]
    for form in forms:
        spec = ", ".join(
            f"SpecRef {{ document: Document::{DOCUMENTS[ref['document']]}, "
            f"section: \"{ref['section']}\", page: {ref['page']} }}"
            for ref in form["spec"]
        )
        verify = ", ".join(f"Evidence::{EVIDENCE[item]}" for item in form["verify"])
        out += [
            "    FormSemantics {",
            f"        name: \"{form['name']}\",",
            f"        family: EncodingFamily::{_rust_family(form['family'])},",
            f"        opcode: {form['opcode']},",
            f"        operation: Operation::{OPERATIONS[form['operation']]},",
            f"        signedness: Signedness::{SIGNEDNESS[form['signedness']]},",
            f"        carry_in: CarryIn::{CARRY_IN[form['carry_in']]},",
            f"        width: {form['width']},",
            f"        shift_count_bits: {form.get(SHIFT_KEY, 0)},",
            f"        shift_amount: {form.get(SHIFT_AMOUNT_KEY, 0)},",
            f"        swap_sources: {'true' if SOURCE_ORDER_KEY in form else 'false'},",
            f"        source_bits: {form.get(SOURCE_BITS_KEY, form['width'])},",
            "        relation: "
            + (f"Some(CompareRelation::{RELATIONS[form[RELATION_KEY]]})," if RELATION_KEY in form else "None,"),
            f"        scc: SccEffect::{SCC[form['scc']]},",
            f"        vcc: MaskEffect::{MASK[form['vcc']]},",
            f"        exec: MaskEffect::{MASK[form['exec']]},",
            f"        wave: Wave::{WAVE[form['wave']]},",
            f"        operands: OperandRule::{OPERAND_RULES[form['operands']]['rust']},",
            f"        spec: &[{spec}],",
            f"        verify: &[{verify}],",
            "    },",
        ]
    out += ["];", ""]
    return "\n".join(out)


def _render_enum(rule_name, forms, snapshot, positions):
    enum = OPERAND_RULES[rule_name]["enum"]
    kind = OPERAND_RULES[rule_name]["kind"]
    table = re.sub(r"(?<!^)(?=[A-Z])", "_", enum).upper() + "_BY_OPCODE"
    family = forms[0]["family"]
    encoding = snapshot["encodings"][_family_encoding(family)]
    mask = encoding["identifier_mask"]
    (offset, size), = dict((name, ranges) for name, ranges in encoding["fields"])["OP"]
    op_mask = ((1 << size) - 1) << offset
    fixed_mask = mask & ~op_mask
    fixed_bits = encoding["identifiers"][0] & fixed_mask
    # Every identifier of the family must share the fixed bits, or the split
    # into an encoding test plus an OP match would change which words match.
    if any(identifier & fixed_mask != fixed_bits for identifier in encoding["identifiers"]) or (
        mask & op_mask != op_mask
    ):
        raise ValueError(f"{family}: identifiers do not split into fixed bits and OP")
    out = [
        "",
        f"/// {family} operations declared in the semantic registry (operand rule `{rule_name}`).",
        "#[derive(Clone, Copy, Debug, Eq, PartialEq)]",
        f"pub enum {enum} {{",
    ]
    for form in forms:
        carry = " plus SCC" if form["carry_in"] == "scc" else ""
        what = f"compare {form[RELATION_KEY]}" if kind == "compare" else form["operation"]
        out += [
            f"    /// `{form['name'].lower()}` ({family} opcode {form['opcode']}): "
            f"{form['signedness']} {what}{carry}, SCC = {form['scc']}.",
            f"    {form['variant']},",
        ]
    by_opcode = {form["opcode"]: form["variant"] for form in forms}
    out += [
        "}",
        "",
        f"/// {enum} by {family} OP field value; `None` for undeclared opcodes.",
        f"const {table}: [Option<{enum}>; {1 << size}] = [",
    ]
    out += [
        f"    Some({enum}::{by_opcode[opcode]})," if opcode in by_opcode else "    None,"
        for opcode in range(1 << size)
    ]
    out += [
        "];",
        "",
        f"impl {enum} {{",
        "    /// Every operation, in registry order.",
        f"    pub const ALL: [Self; {len(forms)}] = [",
    ]
    out += [f"        Self::{form['variant']}," for form in forms]
    out += [
        "    ];",
        "",
        f"    /// Selects the operation from a first instruction word: the {family} encoding",
        f"    /// bits (mask 0x{fixed_mask:09_x}) and then the OP field (bits {offset}-{offset + size - 1}).",
        "    pub const fn from_word(word: u32) -> Option<Self> {",
        f"        if word & 0x{fixed_mask:09_x} != 0x{fixed_bits:09_x} {{",
        "            return None;",
        "        }",
        f"        {table}[((word >> {offset}) & 0x{(1 << size) - 1:x}) as usize]",
        "    }",
        "",
        "    /// Lower-case mnemonic.",
        "    pub const fn mnemonic(self) -> &'static str {",
        "        match self {",
    ]
    out += [f"            Self::{form['variant']} => \"{form['name'].lower()}\"," for form in forms]
    out += [
        "        }",
        "    }",
        "",
        "    /// The validated registry row.",
        "    pub fn semantics(self) -> &'static FormSemantics {",
        "        &FORMS[match self {",
    ]
    out += [f"            Self::{form['variant']} => {positions[form['name']]}," for form in forms]
    out += [
        "        }]",
        "    }",
    ]
    if kind == "unary":
        out += [
            "",
            "    /// Whether the operation writes SCC.",
            "    pub const fn writes_scc(self) -> bool {",
            "        match self {",
        ]
        out += [
            f"            Self::{form['variant']} => {'false' if form['scc'] == 'unchanged' else 'true'},"
            for form in forms
        ]
        out += [
            "        }",
            "    }",
            "",
            "    /// The result and, for forms that write it, the new SCC (`None` leaves SCC",
            "    /// unchanged).",
            "    pub const fn evaluate(self, src: u32) -> (u32, Option<bool>) {",
            "        match self {",
        ]
        for form in forms:
            _, function, _ = PRIMITIVES[(form["operation"], form["signedness"])]
            if form["operation"] == "sext":
                value = f"primitive::{function}(src, {form[SOURCE_BITS_KEY]})"
            else:
                value = f"primitive::{function}(src)"
            if form["scc"] == "unchanged":
                out.append(f"            Self::{form['variant']} => ({value}, None),")
            else:
                out += [
                    f"            Self::{form['variant']} => {{",
                    f"                let (value, scc) = {value};",
                    "                (value, Some(scc))",
                    "            }",
                ]
        out += ["        }", "    }", "}"]
        return out
    if kind == "vector":
        out += [
            "",
            "    /// Assembler name of the 32-bit (VOP2) encoding, as LLVM prints it.",
            "    pub const fn assembly_name(self) -> &'static str {",
            "        match self {",
        ]
        out += [f"            Self::{form['variant']} => \"{form['name'].lower()}_e32\"," for form in forms]
        out += [
            "        }",
            "    }",
            "",
            "    /// The per-lane operation its MSL lowering uses.",
            "    pub const fn lowering(self) -> VectorBinary {",
            "        match self {",
        ]
        out += [
            f"            Self::{form['variant']} => "
            f"VectorBinary::{VECTOR_PRIMITIVES[(form['operation'], form['signedness'])]},"
            for form in forms
        ]
        out += [
            "        }",
            "    }",
            "",
            "    /// Whether the operation reads its sources reversed (`D = op(S1, S0)`).",
            "    pub const fn swaps_sources(self) -> bool {",
            "        match self {",
        ]
        out += [
            f"            Self::{form['variant']} => {'true' if SOURCE_ORDER_KEY in form else 'false'},"
            for form in forms
        ]
        out += ["        }", "    }", "}"]
        return out
    if kind == "compare":
        out += [
            "",
            "    /// The relation tested, with S0 on the left.",
            "    pub const fn relation(self) -> CompareRelation {",
            "        match self {",
        ]
        out += [
            f"            Self::{form['variant']} => CompareRelation::{RELATIONS[form[RELATION_KEY]]},"
            for form in forms
        ]
        out += [
            "        }",
            "    }",
            "",
            "    /// The new SCC: whether the relation holds for S0 and S1.",
            "    pub const fn test(self, src0: u32, src1: u32) -> bool {",
            "        match self {",
        ]
        for form in forms:
            function = COMPARE_PRIMITIVES[(form[RELATION_KEY], form["signedness"])]
            out.append(f"            Self::{form['variant']} => primitive::{function}(src0, src1),")
        out += ["        }", "    }", "}"]
        return out
    out += [
        "",
        "    /// Whether the operation reads SCC (carry-in, borrow-in or select condition).",
        "    pub const fn reads_scc(self) -> bool {",
        "        match self {",
    ]
    out += [
        f"            Self::{form['variant']} => {'true' if form['carry_in'] == 'scc' else 'false'},"
        for form in forms
    ]
    out += [
        "        }",
        "    }",
        "",
        "    /// Whether the operation writes SCC.",
        "    pub const fn writes_scc(self) -> bool {",
        "        match self {",
    ]
    out += [
        f"            Self::{form['variant']} => {'false' if form['scc'] == 'unchanged' else 'true'},"
        for form in forms
    ]
    out += [
        "        }",
        "    }",
        "",
        "    /// The result and, for forms that write it, the new SCC (`None` leaves SCC",
        "    /// unchanged). `scc_in` is the SCC read by forms that consume it (ignored by",
        "    /// the others).",
        "    pub const fn evaluate(self, src0: u32, src1: u32, scc_in: bool) -> (u32, Option<bool>) {",
        "        match self {",
    ]
    for form in forms:
        _, function, takes_carry = PRIMITIVES[(form["operation"], form["signedness"])]
        if takes_carry:
            argument = "scc_in" if form["carry_in"] == "scc" else "false"
            call = f"primitive::{function}(src0, src1, {argument})"
        elif form["operation"] in SHIFTS:
            mask = (1 << form[SHIFT_KEY]) - 1
            call = f"primitive::{function}(src0, src1 & 0x{mask:x})"
        elif form["operation"] == "lshl_add":
            call = f"primitive::{function}(src0, src1, {form[SHIFT_AMOUNT_KEY]})"
        else:
            call = f"primitive::{function}(src0, src1)"
        if form["scc"] == "unchanged":
            out.append(f"            Self::{form['variant']} => ({call}, None),")
        else:
            out += [
                f"            Self::{form['variant']} => {{",
                f"                let (value, scc) = {call};",
                "                (value, Some(scc))",
                "            }",
            ]
    out += ["        }", "    }", "}"]
    return out


def _display(path):
    try:
        return path.resolve().relative_to(ROOT)
    except ValueError:
        return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", nargs="?", choices=("generate", "validate"))
    parser.add_argument("--check", action="store_true", help="fail if the committed file is stale")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--output", type=Path, default=RUST)
    arguments = parser.parse_args()
    if bool(arguments.mode) == arguments.check:
        parser.error("choose exactly one of generate, validate or --check")
    registry = load_registry(arguments.registry)
    snapshot = load_snapshot(arguments.snapshot)
    errors = validate(registry, snapshot)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    count = len(registry["form"])
    if arguments.mode == "validate":
        print(f"registry valid: {count} forms")
        return 0
    text = render(registry, snapshot)
    if arguments.check:
        current = arguments.output.read_text(encoding="utf-8") if arguments.output.exists() else ""
        if current != text:
            print(f"{_display(arguments.output)} is stale; run generate", file=sys.stderr)
            return 1
        print(f"registry valid and generated code current: {count} forms")
        return 0
    arguments.output.write_text(text, encoding="utf-8")
    print(f"wrote {_display(arguments.output)}: {count} forms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
