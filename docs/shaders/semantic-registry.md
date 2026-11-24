# Adding an instruction through the semantic registry

The semantic registry is a data file that declares what each supported gfx1030
instruction form does. A generator checks it and turns it into Rust tables that
the decoder uses. This page explains where entries live, what each field means,
how to check and regenerate the tables, and what the generator does not do for
you.

The registry covers the nine SOP2 forms listed in [SOP2 scalar ALU](scalar-alu.md).

## Files

| File | Role |
| --- | --- |
| `crates/isa/semantics/rdna2.toml` | The registry. Data only: every value comes from a closed vocabulary. |
| `crates/isa/generate_semantics.py` | Validates the registry and writes the Rust tables. Python 3.11 or newer, standard library only. |
| `crates/isa/data/rdna2.jsonl` | Pinned snapshot of the AMD machine-readable ISA XML, used to cross-check identity and operands. |
| `crates/isa/src/semantics_gen.rs` | Generated output. Never edit it by hand. |
| `crates/isa/src/semantics.rs` | The Rust types the generated tables use (`FormSemantics`, `Operation`, `SccEffect` and so on). |
| `crates/isa/src/primitive.rs` | The shared integer primitives that the generated `evaluate` calls. |
| `tests/semantics_generator/` | Unit tests for the generator. |

## Fields

The top of the file holds:

- `schema`: must be `1`.
- `architecture`: must be `"gfx1030"`.
- `source_sha256`: must equal the `sha256` on the `source` line of
  `crates/isa/data/rdna2.jsonl`, so the registry is always checked against the
  snapshot it was written for.
- `[documents.<name>]`: the public documents that spec references cite. Only
  `guide` (the AMD RDNA 2 ISA reference guide) and `xml` (the AMD
  machine-readable ISA) are allowed, and each must match the title and file
  pinned in the generator, so a citation cannot point at another file.
- `[operand_rules.<name>]`: an operand rule shared by a whole family. The
  decoder implements each rule once, and the generator only accepts a rule
  whose declaration matches the decoder's implementation exactly. Today there
  is one rule, `sop2_sgpr`: an SGPR destination `s0`-`s105`, two scalar sources
  (an SGPR, an inline integer -16 to 64, or a literal word) and at most one
  literal. Its `unsupported` list names the encodings the rule rejects, which
  stay unsupported instead of being guessed. The list must name exactly the
  encodings the decoder rejects.

Each `[[form]]` entry has exactly these fields. A missing or unknown field is an
error.

| Field | Meaning | Allowed values |
| --- | --- | --- |
| `name` | XML instruction name, the form's stable identity | upper-case, e.g. `S_ADD_U32` |
| `family` | Encoding family; must match the XML base encoding | e.g. `SOP2` |
| `opcode` | Opcode; must match the XML | non-negative integer |
| `variant` | Rust enum variant of the typed operation | CamelCase, e.g. `AddU32` |
| `operation` | The primitive the form uses | `add`, `sub`, `and`, `or`, `xor`, `andn2`, `orn2`, `nand`, `nor`, `xnor`, `lshl`, `lshr`, `ashr`, `min`, `max`, `cselect`, `bfm`, `mul`, `mul_hi`, `bfe`, `absdiff`, `lshl_add`, `mul24`, `mul_hi24` (vector only), `compare` and the SOP1 unaries |
| `signedness` | How operands are read | `unsigned`, `signed`, `bitwise` (always `bitwise` for the logical operations, `cselect` and `bfm`; never for the arithmetic ones, `add`, `sub`, `min`, `max`, `mul`, `mul_hi`, `bfe`, `absdiff` and `lshl_add`; shifts take their primitive's signedness) |
| `carry_in` | Whether the form reads SCC: carry-in, borrow-in or the select condition | `none`, `scc` (`scc` only for unsigned `add` and `sub`, and required for `cselect`) |
| `width` | Operand and result width in bits | `32` |
| `source_bits` | Sign extension only: the low bits of S0 that are read. The XML must list S0 with that size and format (`FMT_NUM_I8` or `FMT_NUM_I16`) | `8`, `16`; required for `sext`, rejected elsewhere |
| `relation` | Compares only: the relation tested with S0 on the left | `eq`, `lg` (not equal), `gt`, `ge`, `lt`, `le`; required for `compare`, rejected elsewhere |
| `source_order` | Vector forms only: `reversed` when the guide computes `op(S1, S0)` (the XML records no operand order); absent otherwise | `reversed` |
| `shift_amount` | `lshl_add` only: the fixed left shift of S0 before the add | `1` to `4`; required for `lshl_add`, rejected elsewhere |
| `shift_count_bits` | Shifts only: the low bits of S1 used as the count. The generator masks S1 with it, and the XML must list S1 as unsigned | `5` (log2 of `width`); required for shifts, rejected elsewhere |
| `scc` | What the form writes to SCC | `carry_out`, `borrow`, `signed_overflow`, `nonzero`, `relation` (compares only), `selected_src0` (min and max: S0 won the strict compare, so a tie gives 0), `unchanged` |
| `vcc`, `exec` | Effect on VCC and EXEC, stated rather than implied | `unchanged` |
| `wave` | Wave sizes the form is valid in | `any` (`wave32` is in the vocabulary but rejected until the decoder can enforce it) |
| `operands` | Name of an `[operand_rules]` entry for the same family | e.g. `sop2_sgpr` |
| `spec` | References `{ document, section, page }` | at least one `guide` reference with a page above 0 |
| `verify` | Evidence the form needs before it counts as verified | `reference`, `independent`, `metal`, `llvm_encoding` |

The `operation` and `signedness` pair picks one primitive, and that primitive
fixes the `scc` value:

| `operation` / `signedness` | Required `scc` | Primitive | Reads carry-in |
| --- | --- | --- | --- |
| `add` / `unsigned` | `carry_out` | `add_carry_u32` | allowed |
| `sub` / `unsigned` | `borrow` | `sub_borrow_u32` | allowed |
| `add` / `signed` | `signed_overflow` | `add_overflow_i32` | no |
| `sub` / `signed` | `signed_overflow` | `sub_overflow_i32` | no |
| `and`, `or`, `xor`, `andn2`, `orn2`, `nand`, `nor`, `xnor` / `bitwise` | `nonzero` | `and_nonzero_b32`, `or_nonzero_b32`, `xor_nonzero_b32`, `andn2_nonzero_b32`, `orn2_nonzero_b32`, `nand_nonzero_b32`, `nor_nonzero_b32`, `xnor_nonzero_b32` | no |
| `lshl` / `bitwise` | `nonzero` | `lshl_nonzero_b32` | no |
| `lshr` / `unsigned` | `nonzero` | `lshr_nonzero_b32` | no |
| `ashr` / `signed` | `nonzero` | `ashr_nonzero_i32` | no |
| `not`, `bcnt0`, `bcnt1` / `bitwise` | `nonzero` | `not_nonzero_b32`, `bcnt0_nonzero_b32`, `bcnt1_nonzero_b32` | no |
| `brev`, `ff0`, `ff1`, `flbit` / `bitwise` | `unchanged` | `brev_b32`, `ff0_b32`, `ff1_b32`, `flbit_b32` | no |
| `flbit`, `sext` / `signed` | `unchanged` | `flbit_i32`, `sext_i32` | no |
| `abs` / `signed` | `nonzero` | `abs_nonzero_i32` | no |
| `min`, `max` / `signed` or `unsigned` | `selected_src0` | `min_i32`/`min_u32`, `max_i32`/`max_u32` | no |
| `cselect` / `bitwise` | `unchanged` | `cselect_b32` | required |
| `bfm` / `bitwise` | `unchanged` | `bfm_b32` | no |
| `mul` / `signed` | `unchanged` | `mul_lo_i32` | no |
| `mul_hi` / `unsigned` or `signed` | `unchanged` | `mul_hi_u32`, `mul_hi_i32` | no |
| `bfe` / `unsigned` or `signed` | `nonzero` | `bfe_nonzero_u32`, `bfe_nonzero_i32` | no |
| `absdiff` / `signed` | `nonzero` | `absdiff_nonzero_i32` | no |
| `lshl_add` / `unsigned` | `carry_out` | `lshl_add_carry_u32` (with `shift_amount`) | no |
| `compare` / `signed` or `unsigned` | `relation` | `eq_b32`, `ne_b32` (for both signednesses), `gt_i32`/`gt_u32`, `ge_*`, `lt_*`, `le_*` | no |

Vector forms (operand rule `vop2_vgpr`, kind `vector`) use a separate table:
the pair picks the per-lane `VectorBinary` the MSL lowering uses, and `scc`
must be `unchanged` (VOP2 integer forms write no SCC, VCC or EXEC):

| `operation` / `signedness` | `VectorBinary` |
| --- | --- |
| `add` / `unsigned`, `sub` / `unsigned` | `Add`, `Sub` |
| `and`, `or`, `xor`, `xnor` / `bitwise` | `And`, `Or`, `Xor`, `Xnor` |
| `lshl` / `bitwise`, `lshr` / `unsigned`, `ashr` / `signed` | `ShiftLeft`, `LogicalShiftRight`, `ArithmeticShiftRight` |
| `min`, `max` / `signed` or `unsigned` | `MinSigned`, `MinUnsigned`, `MaxSigned`, `MaxUnsigned` |
| `mul24`, `mul_hi24` / `signed` or `unsigned` | `Mul24Signed`, `Mul24Unsigned`, `MulHi24Signed`, `MulHi24Unsigned` |

No registry form uses `And`, `Or`, `Xor` or the three shifts yet. The
hand-written VGPR-only forms `v_and_b32`, `v_or_b32`, `v_xor_b32`,
`v_lshlrev_b32`, `v_lshrrev_b32` and `v_ashrrev_i32` already lower through
these templates. Declaring them here is waiting on their guide page
citations, which have not been checked.

A compare must use an operand rule of the compare kind (`sopc_scalar`: two
sources, no destination), a unary operation one of the unary kind (`sop1_sgpr`:
one source and an SGPR destination), and a two-source ALU operation
`sop2_sgpr`.

Behaviour is chosen only from these declared fields, never from the mnemonic.

## What the validator checks

`validate` reports every error it finds, not only the first (a form with a missing field is not checked further). Besides the
vocabularies above, it rejects:

- contradictions: a declared `scc` or `carry_in` that the primitive does not
  have, `bitwise` on an arithmetic operation or the reverse;
- duplicates: the same `name`, the same `(family, opcode)`, or the same
  `variant` twice in one generated enum;
- XML disagreements: a family with no XML encoding, a form with no default XML
  variant, a different opcode, an SCC input or output the XML does not show (or
  that the declaration omits), an operand width other than `width`, a signed
  declaration on unsigned XML formats or the reverse (a shift count or the `bfe`
  field descriptor must be `FMT_NUM_U32` whatever the signedness; it is S1, or S0
  for a vector form with `source_order = "reversed"` such as `v_lshlrev_b32`), an operand direction that
  differs, and XML operands the operand rule does not cover;
- references: a spec citing an undeclared document, a bad section or page, or
  no guide page at all;
- operand rules the decoder does not implement, or whose declaration differs
  from the implemented one;
- wrong types: a float width, a boolean schema, an opcode or page outside
  0-65535, or a reserved variant name such as `Self`.

CI runs `generate_semantics.py --check`, `scripts/audit/semantic_coverage.py
--check` and both Python test suites next to the existing
`generate_rdna2_meta.py --check`. Both scripts stop with a clear message on
Python older than 3.11.

## What the generator does not do

- **The XML gives identity and encodings only.** The snapshot confirms the
  name, family, opcode, operand fields, widths, number formats and whether SCC
  is read or written. It says nothing about the arithmetic. Behaviour comes from
  the AMD guide, cited in `spec`, through the closed primitive table.
- **It does not invent behaviour.** A form whose operation, signedness or
  operand rule is not already implemented is rejected. Adding a new operation
  means hand-writing a primitive in `primitive.rs`, a variant in
  `semantics.rs` and a row in the generator's vocabulary tables, reviewed like
  any other code.
- **Undeclared forms stay unsupported.** The generated `from_word` returns
  `None` for every opcode not in the registry, and the decoder reports
  `UnsupportedOpcode`. Nothing falls back to a guess or a no-op.
- **It does not write tests or reference semantics.** The reference
  interpreter (`crates/ref`) and the independent interpreter
  (`tests/validation/independent_ref.rs`) never read the registry or the
  generated code, and their known-answer tests are written by hand from the
  guide. That independence is what lets them catch a wrong registry entry; keep
  it when you add a form. Agreement with these references is agreement with a
  reading of the public AMD guide, not with Xbox hardware.

## Commands

Run these from the repository root. `python3` must be 3.11 or newer.

```sh
# Check the registry only.
python3 crates/isa/generate_semantics.py validate

# Validate, then rewrite crates/isa/src/semantics_gen.rs.
python3 crates/isa/generate_semantics.py generate

# Validate and regenerate in memory; exit 1 if the committed file is stale. Never writes.
python3 crates/isa/generate_semantics.py --check

# Generator unit tests.
python3 -m unittest discover -s tests/semantics_generator

# The XML metadata tables the registry is cross-checked against are still current.
python3 crates/isa/generate_rdna2_meta.py --check

# Rust tests of the generated tables and the decoder that uses them.
cargo test --locked -p xenora-isa -p xenora-decoder

# Independent CPU validation.
cargo test --locked -p xenora-semantic-review

# Native GPU correctness on Apple silicon.
scripts/test-gpu.sh
```

The LLVM encoding corpus check needs an LLVM build with the AMDGPU target:

```sh
python3 crates/isa/tests/llvm/corpus.py --llvm-bin /path/to/llvm/bin --check
```

## Worked example: `s_addc_u32`

`s_addc_u32` adds two 32-bit sources plus SCC and writes the carry-out of
bit 31 to SCC (guide section 12.1, page 102; SCC rules in section 5.3, page 33).
Its registry entry is:

```toml
[[form]]
name = "S_ADDC_U32"
family = "SOP2"
opcode = 4
variant = "AddcU32"
operation = "add"
signedness = "unsigned"
carry_in = "scc"
width = 32
scc = "carry_out"
vcc = "unchanged"
exec = "unchanged"
wave = "any"
operands = "sop2_sgpr"
spec = [
  { document = "guide", section = "12.1", page = 102 },
  { document = "guide", section = "5.3", page = 33 },
]
verify = ["reference", "independent", "metal", "llvm_encoding"]
```

How each field is checked:

1. The XML snapshot has a default `ENC_SOP2` variant named `S_ADDC_U32` with
   opcode 4, so `name`, `family` and `opcode` agree.
2. That XML variant reads and writes SCC, which matches `carry_in = "scc"` and
   an `scc` other than `unchanged`.
3. `add` / `unsigned` selects `add_carry_u32`, which requires
   `scc = "carry_out"` and accepts a carry-in.
4. The XML `SDST`, `SSRC0` and `SSRC1` operands are 32-bit `FMT_NUM_U32`
   fields, which matches `width = 32`, `unsigned` and the `sop2_sgpr` rule.

From this entry the generator emits, in `semantics_gen.rs`, the `AddcU32`
variant of `ScalarAluOp`, its `from_word` arm (`0x8200_0000`), the mnemonic
`s_addc_u32`, `reads_scc() == true`, an `evaluate` arm that calls
`primitive::add_carry_u32(src0, src1, carry_in)`, and a `FormSemantics` row.

To add a form the same way:

1. Confirm its behaviour in the AMD guide and note the section and page.
2. Check that an existing primitive and operand rule cover it. If not, the
   form needs hand-written code first (see above).
3. Add a `[[form]]` entry and run `validate` until it reports no errors.
4. Run `generate` and commit the registry and `semantics_gen.rs` together.
5. Add known-answer tests to the reference and independent interpreters by
   hand, then run every command in [Commands](#commands).
