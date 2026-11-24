// SPDX-License-Identifier: GPL-3.0-or-later

//! Verified compilation for Xenora's synthetic compute and graphics ABIs.

use std::fmt::{self, Write};

mod graphics;
pub use graphics::{PIXEL_ENTRY_POINT, VERTEX_ENTRY_POINT, compile_graphics};

pub use xenora_decoder::Architecture;
use xenora_decoder::{
    DecodeError, DecodedInstruction, Instruction, ScalarSource, VectorBinary, VectorCompareOp,
    VectorSource,
};

/// SGPR index of `vcc_lo`.
const VCC_LO: usize = 106;
/// SGPR index of `vcc_hi`.
const VCC_HI: usize = 107;

/// Entry point of all kernels produced by this compiler.
pub const ENTRY_POINT: &str = "xenora_compute";
const MAX_CODE_BYTES: usize = 1024;

#[derive(Debug)]
pub struct CompiledShader {
    pub disassembly: String,
    pub ir: String,
    pub msl: String,
}

#[derive(Debug)]
pub enum CompileError {
    InvalidLength {
        bytes: usize,
    },
    Decode(DecodeError),
    GraphicsStage {
        stage: &'static str,
        error: Box<CompileError>,
    },
    GraphicsLink {
        reason: &'static str,
    },
    Verify {
        offset: usize,
        reason: &'static str,
    },
}

impl fmt::Display for CompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { bytes } => write!(
                formatter,
                "fixture length must be a nonzero multiple of four up to {MAX_CODE_BYTES} bytes, got {bytes}"
            ),
            Self::Decode(error) => error.fmt(formatter),
            Self::GraphicsStage { stage, error } => write!(formatter, "{stage} shader: {error}"),
            Self::GraphicsLink { reason } => write!(formatter, "graphics linking: {reason}"),
            Self::Verify { offset, reason } => {
                write!(
                    formatter,
                    "shader verification at byte 0x{offset:x}: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for CompileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Decode(error) => Some(error),
            Self::GraphicsStage { error, .. } => Some(error),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
struct Value(usize);

impl fmt::Display for Value {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "t{}", self.0)
    }
}

#[derive(Clone, Copy)]
enum Op {
    Load,
    Constant(u32),
    Binary(Binary, Value, Value),
    MergeLanes {
        mask: u32,
        active: Value,
        inactive: Value,
    },
    Store(Value),
    ScalarBranch {
        target: usize,
        taken: bool,
    },
    /// Wave32 VOPC result: bit n is EXEC[n] & (lane n exists) & test(lane n).
    /// The value is wave-uniform; every thread computes it for its own wave.
    LaneMask {
        op: VectorCompareOp,
        src0: Value,
        src1: Value,
        exec: u32,
    },
    /// `s_cbranch_vccz` on a lane-derived VCC_LO (IR annotation only).
    RuntimeBranch {
        mask: Value,
        target: usize,
    },
    /// Join after a runtime `s_cbranch_vccz`: the fall-through value when the
    /// mask is nonzero (branch not taken), else the value from before the branch.
    JoinVccz {
        mask: Value,
        fallthrough: Value,
        taken: Value,
    },
}

impl Op {
    fn operands(self) -> Vec<Value> {
        match self {
            Self::Load | Self::Constant(_) | Self::ScalarBranch { .. } => Vec::new(),
            Self::Binary(_, left, right) => vec![left, right],
            Self::MergeLanes {
                active, inactive, ..
            } => vec![active, inactive],
            Self::Store(value) => vec![value],
            Self::LaneMask { src0, src1, .. } => vec![src0, src1],
            Self::RuntimeBranch { mask, .. } => vec![mask],
            Self::JoinVccz {
                mask,
                fallthrough,
                taken,
            } => vec![mask, fallthrough, taken],
        }
    }

    /// The MSL expression of a per-thread value, with operand names, the
    /// dispatch index and the lane given by the caller.
    fn expression(self, name: &dyn Fn(Value) -> String, index: &str, lane: &str) -> String {
        match self {
            Self::Load => format!("input[{index}]"),
            Self::Constant(value) => format!("0x{value:08x}u"),
            Self::Binary(operator, left, right) => {
                let (left, right) = (name(left), name(right));
                let (_, symbol) = operator.operator();
                match operator {
                    Binary::FloatAdd | Binary::FloatMul => format!(
                        "as_type<uint>(as_type<float>({left}) {symbol} as_type<float>({right}))"
                    ),
                    Binary::Vector(operation) => vector_expression(operation, &left, &right),
                    Binary::Add => format!("{left} {symbol} {right}"),
                }
            }
            Self::MergeLanes {
                mask,
                active,
                inactive,
            } => format!(
                "((0x{mask:08x}u >> {lane}) & 1u) != 0u ? {} : {}",
                name(active),
                name(inactive)
            ),
            Self::JoinVccz {
                mask,
                fallthrough,
                taken,
            } => format!(
                "{} != 0u ? {} : {}",
                name(mask),
                name(fallthrough),
                name(taken)
            ),
            Self::Store(_)
            | Self::ScalarBranch { .. }
            | Self::LaneMask { .. }
            | Self::RuntimeBranch { .. } => unreachable!("not a per-thread expression"),
        }
    }
}

fn compare_expression(op: VectorCompareOp, left: &str, right: &str) -> String {
    use xenora_decoder::CompareRelation;
    let symbol = match op.relation {
        CompareRelation::Less => "<",
        CompareRelation::Equal => "==",
        CompareRelation::LessOrEqual => "<=",
        CompareRelation::Greater => ">",
        CompareRelation::NotEqual => "!=",
        CompareRelation::GreaterOrEqual => ">=",
    };
    if op.signed {
        format!("as_type<int>({left}) {symbol} as_type<int>({right})")
    } else {
        format!("{left} {symbol} {right}")
    }
}

#[derive(Clone, Copy)]
enum Binary {
    Add,
    FloatAdd,
    FloatMul,
    /// A per-lane VOP2 integer operation, lowered by its typed template. The
    /// hand-written bitwise and shift forms use these templates too.
    Vector(VectorBinary),
}

impl Binary {
    fn operator(self) -> (&'static str, &'static str) {
        match self {
            Self::Add => ("add.u32", "+"),
            Self::FloatAdd => ("add.f32", "+"),
            Self::FloatMul => ("mul.f32", "*"),
            Self::Vector(operation) => (
                match operation {
                    VectorBinary::Add => "add.u32",
                    VectorBinary::Sub => "sub.u32",
                    VectorBinary::And => "and.b32",
                    VectorBinary::Or => "or.b32",
                    VectorBinary::Xor => "xor.b32",
                    VectorBinary::Xnor => "xnor.b32",
                    VectorBinary::ShiftLeft => "lshl.b32",
                    VectorBinary::LogicalShiftRight => "lshr.b32",
                    VectorBinary::ArithmeticShiftRight => "ashr.i32",
                    VectorBinary::MinSigned => "min.i32",
                    VectorBinary::MinUnsigned => "min.u32",
                    VectorBinary::MaxSigned => "max.i32",
                    VectorBinary::MaxUnsigned => "max.u32",
                    VectorBinary::Mul24Signed => "mul24.i32",
                    VectorBinary::Mul24Unsigned => "mul24.u32",
                    VectorBinary::MulHi24Signed => "mulhi24.i32",
                    VectorBinary::MulHi24Unsigned => "mulhi24.u32",
                },
                "",
            ),
        }
    }
}

/// The MSL template of each per-lane VOP2 integer operation. Integer
/// arithmetic is on `uint` (wrapping); signed forms reinterpret with `as_type`,
/// and the 24-bit forms sign-extend bit 23 with a shift pair or mask to 24 bits
/// first. Shifts read only the low five bits of the count (`right`), the
/// five-bit count the reference interpreters also apply.
fn vector_expression(operation: VectorBinary, left: &str, right: &str) -> String {
    let int = |value: &str| format!("as_type<int>({value})");
    let i24 = |value: &str| format!("(as_type<int>({value} << 8u) >> 8)");
    let u24 = |value: &str| format!("({value} & 0x00ffffffu)");
    match operation {
        VectorBinary::Add => format!("{left} + {right}"),
        VectorBinary::Sub => format!("{left} - {right}"),
        VectorBinary::And => format!("{left} & {right}"),
        VectorBinary::Or => format!("{left} | {right}"),
        VectorBinary::Xor => format!("{left} ^ {right}"),
        VectorBinary::Xnor => format!("~({left} ^ {right})"),
        VectorBinary::ShiftLeft => format!("{left} << ({right} & 31u)"),
        VectorBinary::LogicalShiftRight => format!("{left} >> ({right} & 31u)"),
        VectorBinary::ArithmeticShiftRight => {
            format!("as_type<uint>({} >> ({right} & 31u))", int(left))
        }
        VectorBinary::MinSigned => format!("as_type<uint>(min({}, {}))", int(left), int(right)),
        VectorBinary::MinUnsigned => format!("min({left}, {right})"),
        VectorBinary::MaxSigned => format!("as_type<uint>(max({}, {}))", int(left), int(right)),
        VectorBinary::MaxUnsigned => format!("max({left}, {right})"),
        // The low 32 bits of a product do not depend on signedness, so the
        // sign-extended factors multiply as wrapping uint.
        VectorBinary::Mul24Signed => format!(
            "as_type<uint>({}) * as_type<uint>({})",
            i24(left),
            i24(right)
        ),
        VectorBinary::Mul24Unsigned => format!("{} * {}", u24(left), u24(right)),
        VectorBinary::MulHi24Signed => {
            format!("as_type<uint>(mulhi({}, {}))", i24(left), i24(right))
        }
        VectorBinary::MulHi24Unsigned => format!("mulhi({}, {})", u24(left), u24(right)),
    }
}

// Each bit is either known zero, known one or unconstrained.
#[derive(Clone, Copy)]
struct KnownBits {
    zero: u32,
    one: u32,
    values: Option<[u32; 2]>,
}

impl KnownBits {
    const UNKNOWN: Self = Self {
        zero: 0,
        one: 0,
        values: None,
    };

    fn constant(value: u32) -> Self {
        Self {
            zero: !value,
            one: value,
            values: Some([value; 2]),
        }
    }

    fn and(self, other: Self) -> Self {
        Self {
            zero: self.zero | other.zero,
            one: self.one & other.one,
            values: None,
        }
    }

    fn or(self, other: Self) -> Self {
        Self {
            zero: self.zero & other.zero,
            one: self.one | other.one,
            values: None,
        }
    }

    fn xor(self, other: Self) -> Self {
        Self {
            zero: (self.zero & other.zero) | (self.one & other.one),
            one: (self.zero & other.one) | (self.one & other.zero),
            values: None,
        }
    }

    fn positive_normal_candidates(self) -> Option<[f32; 2]> {
        let values = match self.values {
            Some(values) => values,
            None => {
                let unknown = !(self.zero | self.one);
                if unknown.count_ones() > 1 {
                    return None;
                }
                [self.one, self.one | unknown]
            }
        };
        let candidates = values.map(f32::from_bits);
        candidates
            .iter()
            .all(|n| n.is_normal() && n.is_sign_positive())
            .then_some(candidates)
    }
}

fn exact_float_result(operator: Binary, left: KnownBits, right: KnownBits) -> Option<KnownBits> {
    let left = left.positive_normal_candidates()?;
    let right = right.positive_normal_candidates()?;
    let mut zero = u32::MAX;
    let mut one = u32::MAX;
    let mut results = Vec::with_capacity(4);
    for a in left.map(f64::from) {
        for b in right.map(f64::from) {
            let exact = match operator {
                Binary::FloatAdd => a + b,
                Binary::FloatMul => a * b,
                _ => return None,
            };
            let value = exact as f32;
            if !value.is_normal() || f64::from(value) != exact {
                return None;
            }
            // Binary64 can round away the smaller operand of a widely separated
            // binary32 sum. Require that subtraction recovers the smaller input.
            if matches!(operator, Binary::FloatAdd) && exact - a.max(b) != a.min(b) {
                return None;
            }
            let bits = value.to_bits();
            zero &= !bits;
            one &= bits;
            if !results.contains(&bits) {
                results.push(bits);
            }
        }
    }
    let values = match results.as_slice() {
        [only] => Some([*only; 2]),
        [first, second] => Some([*first, *second]),
        _ => None,
    };
    Some(KnownBits { zero, one, values })
}

/// Compiles explicitly tagged raw machine code to a synthetic Metal compute kernel.
///
/// Bind equal-length disjoint input and output word buffers and dispatch one
/// invocation per word in a one-dimensional grid.
pub fn compile(architecture: Architecture, bytes: &[u8]) -> Result<CompiledShader, CompileError> {
    let decoded = decode_fixture(architecture, bytes)?;
    let operations = lift(&decoded, bytes.len())?;

    let disassembly = decoded.iter().map(|line| format!("{line}\n")).collect();
    let mut ir = String::new();
    let has_mask = operations
        .iter()
        .any(|(_, operation)| matches!(operation, Op::LaneMask { .. }));
    let grid = if has_mask {
        ",\n    uint grid [[threads_per_grid]]"
    } else {
        ""
    };
    let mut msl = format!(
        "#include <metal_stdlib>\nusing namespace metal;\nkernel void {ENTRY_POINT}(\n    device const uint* input [[buffer(0)]],\n    device uint* output [[buffer(1)]],\n    uint index [[thread_position_in_grid]]{grid}) {{\n"
    );
    if operations.iter().any(|(_, operation)| {
        matches!(
            operation,
            Op::Binary(Binary::FloatAdd | Binary::FloatMul, _, _)
        )
    }) {
        msl.push_str("    #pragma clang fp contract(off)\n    #pragma clang fp reassociate(off)\n");
    }
    // Wave-uniform values: identical in every lane of an RDNA wave.
    let mut uniform = Vec::with_capacity(operations.len());
    for (index, &(offset, operation)) in operations.iter().enumerate() {
        let dest = Value(index);
        uniform.push(match operation {
            Op::Constant(_) | Op::LaneMask { .. } => true,
            Op::Load | Op::MergeLanes { .. } => false,
            _ => operation.operands().iter().all(|value| uniform[value.0]),
        });
        let thread = |value: Value| value.to_string();
        let (step, statement) = match operation {
            Op::Load => (
                format!("{dest} = load.u32 input[index]"),
                format!(
                    "uint {dest} = {};",
                    operation.expression(&thread, "index", "")
                ),
            ),
            Op::Constant(value) => (
                format!("{dest} = const.u32 0x{value:08x}"),
                format!("uint {dest} = {};", operation.expression(&thread, "", "")),
            ),
            Op::Binary(operator, left, right) => (
                format!("{dest} = {} {left}, {right}", operator.operator().0),
                format!("uint {dest} = {};", operation.expression(&thread, "", "")),
            ),
            Op::ScalarBranch { target, taken } => {
                let _ = writeln!(
                    ir,
                    "{offset:04x}: branch.uniform 0x{target:04x} taken={taken}"
                );
                continue;
            }
            Op::RuntimeBranch { mask, target } => {
                let _ = writeln!(ir, "{offset:04x}: branch.vccz {mask} -> 0x{target:04x}");
                continue;
            }
            Op::MergeLanes {
                mask,
                active,
                inactive,
            } => (
                format!("{dest} = merge.lanes mask=0x{mask:08x}, {active}, {inactive}"),
                format!(
                    "uint {dest} = {};",
                    operation.expression(&thread, "", "(index & 31u)")
                ),
            ),
            Op::JoinVccz {
                mask,
                fallthrough,
                taken,
            } => (
                format!("{dest} = join.vccz {mask}, {fallthrough}, {taken}"),
                format!("uint {dest} = {};", operation.expression(&thread, "", "")),
            ),
            Op::LaneMask {
                op,
                src0,
                src1,
                exec,
            } => {
                let _ = writeln!(
                    ir,
                    "{offset:04x}: {dest} = wave.mask {} {src0}, {src1} exec=0x{exec:08x}",
                    op.mnemonic()
                );
                emit_lane_mask(&mut msl, &operations, &uniform, index, op, src0, src1, exec);
                continue;
            }
            Op::Store(value) => (
                format!("store.u32 output[index], {value}"),
                format!("output[index] = {value};"),
            ),
        };
        // Writing to a String is infallible.
        let _ = writeln!(ir, "{offset:04x}: {step}");
        let _ = writeln!(msl, "    {statement}");
    }
    msl.push_str("}\n");
    Ok(CompiledShader {
        disassembly,
        ir,
        msl,
    })
}

/// Emit a Wave32 compare mask without any cross-thread operation: the thread
/// recomputes the lane-dependent operands of each lane of its own RDNA wave
/// (dispatch indices `(index & ~31) + n`) from those lanes' input words.
/// Absent lanes of a final partial wave contribute 0.
#[allow(clippy::too_many_arguments)]
fn emit_lane_mask(
    msl: &mut String,
    operations: &[(usize, Op)],
    uniform: &[bool],
    index: usize,
    op: VectorCompareOp,
    src0: Value,
    src1: Value,
    exec: u32,
) {
    let mut cone = vec![false; index];
    let mut pending = vec![src0, src1];
    while let Some(value) = pending.pop() {
        if uniform[value.0] || cone[value.0] {
            continue;
        }
        cone[value.0] = true;
        pending.extend(operations[value.0].1.operands());
    }
    let name = |value: Value| {
        if cone[value.0] {
            format!("l{}", value.0)
        } else {
            value.to_string()
        }
    };
    let dest = Value(index);
    let _ = writeln!(msl, "    uint {dest} = 0u;");
    let _ = writeln!(
        msl,
        "    for (uint lane = 0u; lane < 32u; ++lane) {{\n        uint lane_index = (index & ~31u) + lane;\n        if (lane_index >= grid) {{ break; }}"
    );
    for (position, &(_, operation)) in operations[..index].iter().enumerate() {
        if cone[position] {
            let _ = writeln!(
                msl,
                "        uint l{position} = {};",
                operation.expression(&name, "lane_index", "lane")
            );
        }
    }
    let _ = writeln!(
        msl,
        "        uint pass = ({}) ? 1u : 0u;\n        {dest} |= ((0x{exec:08x}u >> lane) & pass) << lane;\n    }}",
        compare_expression(op, &name(src0), &name(src1))
    );
}

fn decode_fixture(
    architecture: Architecture,
    bytes: &[u8],
) -> Result<Vec<DecodedInstruction>, CompileError> {
    if bytes.is_empty() || bytes.len() > MAX_CODE_BYTES || !bytes.len().is_multiple_of(4) {
        return Err(CompileError::InvalidLength { bytes: bytes.len() });
    }
    xenora_decoder::decode(architecture, bytes).map_err(CompileError::Decode)
}

fn lift(decoded: &[DecodedInstruction], code_len: usize) -> Result<Vec<(usize, Op)>, CompileError> {
    let mut operations = Vec::new();
    let mut registers = [None::<Value>; 256];
    let mut facts = Vec::<KnownBits>::new();
    let mut pending_load = None;
    let mut stored = false;
    let mut store_pending = false;
    let mut scalars = [None; 128];
    let mut exec = u32::MAX; // The fixture starts with full Wave32 EXEC.
    scalars[126] = Some(exec);
    let mut scc = None;
    let mut skip_before = 0;
    // Set while VCC_LO holds a lane-derived compare mask (scalars[VCC_LO] is then None).
    let mut lane_vcc: Option<Value> = None;
    // An open runtime `s_cbranch_vccz` region: its end, mask and the VGPRs at the branch.
    let mut region: Option<(usize, Value, [Option<Value>; 256])> = None;

    // Every decoded instruction must satisfy the fixture layout and binding
    // contract, even when a known branch skips it.
    for (index, instruction) in decoded.iter().enumerate() {
        let offset = instruction.offset;
        let reason = match instruction.instruction {
            Instruction::Branch { target }
            | Instruction::BranchSccZero { target }
            | Instruction::BranchExecZero { target }
            | Instruction::BranchVccZero { target }
                if target <= offset
                    || decoded
                        .binary_search_by_key(&target, |instruction| instruction.offset)
                        .is_err() =>
            {
                "branch target must be a forward instruction boundary"
            }
            Instruction::BufferLoadDword {
                index, resource, ..
            } if (index, resource) != (0, 0) => "unsupported resource or buffer index",
            Instruction::BufferStoreDword {
                index, resource, ..
            } if (index, resource) != (0, 4) => "unsupported resource or buffer index",
            Instruction::Export { .. }
            | Instruction::Interpolate { .. }
            | Instruction::ImageSampleLz { .. } => {
                "graphics instruction is unsupported in a compute fixture"
            }
            Instruction::EndProgram if index + 1 != decoded.len() => {
                return Err(CompileError::Verify {
                    offset: decoded[index + 1].offset,
                    reason: "instruction follows program termination",
                });
            }
            _ => continue,
        };
        return Err(CompileError::Verify { offset, reason });
    }

    for instruction in decoded {
        let offset = instruction.offset;
        let error = |reason| CompileError::Verify { offset, reason };
        if let Some((end, mask, before)) = region
            && offset >= end
        {
            region = None;
            for slot in 0..registers.len() {
                if let (Some(fallthrough), Some(taken)) = (registers[slot], before[slot])
                    && fallthrough.0 != taken.0
                {
                    let next = Value(operations.len());
                    operations.push((
                        end,
                        Op::JoinVccz {
                            mask,
                            fallthrough,
                            taken,
                        },
                    ));
                    let (left, right) = (facts[fallthrough.0], facts[taken.0]);
                    // Keep up to two candidate values when both edges have them.
                    let values = match (left.values, right.values) {
                        (Some([a, b]), Some([c, d])) => {
                            let mut union = vec![a, b, c, d];
                            union.sort_unstable();
                            union.dedup();
                            match union.as_slice() {
                                [only] => Some([*only; 2]),
                                [first, second] => Some([*first, *second]),
                                _ => None,
                            }
                        }
                        _ => None,
                    };
                    facts.push(KnownBits {
                        zero: left.zero & right.zero,
                        one: left.one & right.one,
                        values,
                    });
                    registers[slot] = Some(next);
                }
            }
        }
        // docs/shaders/vector-compare.md section 6: checked whatever EXEC is.
        if let Some((_, _, before)) = &region {
            let dst = match instruction.instruction {
                Instruction::MoveB32 { dst, .. }
                | Instruction::AddNoCarryU32 { dst, .. }
                | Instruction::VectorLshlrevB32 { dst, .. }
                | Instruction::VectorLshrrevB32 { dst, .. }
                | Instruction::VectorAshrrevI32 { dst, .. }
                | Instruction::VectorAndB32 { dst, .. }
                | Instruction::VectorOrB32 { dst, .. }
                | Instruction::VectorXorB32 { dst, .. } => dst,
                _ => {
                    return Err(error(
                        "code skipped by a runtime s_cbranch_vccz may contain only v_mov_b32, v_add_nc_u32_e64 and the hand-written integer VOP2 forms",
                    ));
                }
            };
            if before[usize::from(dst)].is_none() {
                return Err(error(
                    "a VGPR written under a runtime s_cbranch_vccz must be initialized before the branch",
                ));
            }
        }
        let read = |register: u8| {
            registers[usize::from(register)].ok_or_else(|| {
                error("source register is uninitialized, pending a load wait, or reserved for indexing")
            })
        };
        if offset < skip_before {
            continue;
        }
        let read_scalar = |source| match source {
            ScalarSource::Immediate(value) => Ok(value),
            ScalarSource::Register(register) => scalars
                .get(usize::from(register))
                .copied()
                .flatten()
                .ok_or_else(|| error("scalar source has no known value")),
        };
        // Scalar instructions and waits still execute when all lanes are inactive.
        if exec == 0 {
            match instruction.instruction {
                Instruction::MoveB32 { .. }
                | Instruction::AddNoCarryU32 { .. }
                | Instruction::VectorLshlrevB32 { .. }
                | Instruction::VectorLshrrevB32 { .. }
                | Instruction::VectorAshrrevI32 { .. }
                | Instruction::VectorAndB32 { .. }
                | Instruction::VectorOrB32 { .. }
                | Instruction::VectorXorB32 { .. }
                | Instruction::VectorAddF32 { .. }
                | Instruction::VectorMulF32 { .. }
                | Instruction::VectorAlu { .. } => continue,
                _ => {}
            }
        }
        if exec != u32::MAX
            && matches!(
                instruction.instruction,
                Instruction::BufferLoadDword { .. } | Instruction::BufferStoreDword { .. }
            )
        {
            return Err(error("vector memory requires full Wave32 EXEC"));
        }
        let next = Value(operations.len());
        let (dest, operation) = match instruction.instruction {
            Instruction::BufferLoadDword { data, .. } => {
                let slot = usize::from(data);
                if data == 0 || pending_load.is_some() {
                    return Err(error(
                        "load writes the index register or another load is outstanding",
                    ));
                }
                registers[slot] = None;
                pending_load = Some((slot, next));
                (None, Op::Load)
            }
            Instruction::BufferStoreDword { data, .. } => {
                if stored {
                    return Err(error("only one output store is supported"));
                }
                stored = true;
                store_pending = true;
                (None, Op::Store(read(data)?))
            }
            Instruction::WaitLoad => {
                if let Some((register, value)) = pending_load.take() {
                    registers[register] = Some(value);
                }
                continue;
            }
            Instruction::WaitStore => {
                store_pending = false;
                continue;
            }
            Instruction::MoveB32 {
                dst,
                src: VectorSource::Register(src),
            } => {
                let value = read(src)?;
                let slot = usize::from(dst);
                if dst == 0 || pending_load.is_some_and(|(register, _)| register == slot) {
                    return Err(error(
                        "move writes the index register or replaces a pending load",
                    ));
                }
                registers[slot] = Some(merge_lanes(
                    &mut operations,
                    &mut facts,
                    offset,
                    exec,
                    value,
                    registers[slot],
                )?);
                continue;
            }
            Instruction::MoveB32 {
                dst,
                src: VectorSource::Immediate(value),
            } => (Some(dst), Op::Constant(value)),
            Instruction::MoveB32 {
                dst,
                src: VectorSource::Scalar(register),
            } if usize::from(register) == VCC_LO
                && let Some(mask) = lane_vcc =>
            {
                let slot = usize::from(dst);
                if dst == 0 || pending_load.is_some_and(|(register, _)| register == slot) {
                    return Err(error(
                        "move writes the index register or replaces a pending load",
                    ));
                }
                registers[slot] = Some(merge_lanes(
                    &mut operations,
                    &mut facts,
                    offset,
                    exec,
                    mask,
                    registers[slot],
                )?);
                continue;
            }
            // SGPR values are wave-uniform and known during translation.
            Instruction::MoveB32 {
                dst,
                src: VectorSource::Scalar(register),
            } => (
                Some(dst),
                Op::Constant(read_scalar(ScalarSource::Register(register))?),
            ),
            Instruction::AddNoCarryU32 { dst, src0, src1 } => {
                (Some(dst), Op::Binary(Binary::Add, read(src0)?, read(src1)?))
            }
            // The hand-written VGPR-only VOP2 forms lower through the same
            // VectorBinary templates as registry forms. The reversed shifts
            // shift S1 by the count in S0.
            Instruction::VectorLshlrevB32 { dst, src0, src1 } => (
                Some(dst),
                Op::Binary(
                    Binary::Vector(VectorBinary::ShiftLeft),
                    read(src1)?,
                    read(src0)?,
                ),
            ),
            Instruction::VectorAshrrevI32 { dst, src0, src1 } => (
                Some(dst),
                Op::Binary(
                    Binary::Vector(VectorBinary::ArithmeticShiftRight),
                    read(src1)?,
                    read(src0)?,
                ),
            ),
            Instruction::VectorLshrrevB32 { dst, src0, src1 } => (
                Some(dst),
                Op::Binary(
                    Binary::Vector(VectorBinary::LogicalShiftRight),
                    read(src1)?,
                    read(src0)?,
                ),
            ),
            Instruction::VectorAndB32 { dst, src0, src1 } => (
                Some(dst),
                Op::Binary(Binary::Vector(VectorBinary::And), read(src0)?, read(src1)?),
            ),
            Instruction::VectorOrB32 { dst, src0, src1 } => (
                Some(dst),
                Op::Binary(Binary::Vector(VectorBinary::Or), read(src0)?, read(src1)?),
            ),
            Instruction::VectorXorB32 { dst, src0, src1 } => (
                Some(dst),
                Op::Binary(Binary::Vector(VectorBinary::Xor), read(src0)?, read(src1)?),
            ),
            Instruction::ScalarMoveB32 { dst, src } => {
                let value = read_scalar(src)?;
                if dst < 8 {
                    return Err(error("scalar move would modify a resource descriptor"));
                }
                if dst == 126 {
                    exec = value;
                }
                if usize::from(dst) == VCC_LO {
                    lane_vcc = None;
                }
                let slot = scalars
                    .get_mut(usize::from(dst))
                    .ok_or_else(|| error("scalar destination exceeds the fixture register file"))?;
                *slot = Some(value);
                continue;
            }
            Instruction::ScalarUnary { op, dst, src } => {
                let src = read_scalar(src)?;
                if dst < 8 {
                    return Err(error("scalar unary would modify a resource descriptor"));
                }
                let (value, flag) = op.evaluate(src);
                let slot = scalars
                    .get_mut(usize::from(dst))
                    .ok_or_else(|| error("scalar destination exceeds the fixture register file"))?;
                *slot = Some(value);
                if let Some(flag) = flag {
                    scc = Some(flag);
                }
                continue;
            }
            Instruction::ScalarAlu {
                op,
                dst,
                src0,
                src1,
            } => {
                let (src0, src1) = (read_scalar(src0)?, read_scalar(src1)?);
                let scc_in = if op.reads_scc() {
                    scc.ok_or_else(|| error("SOP2 SCC input reads uninitialized SCC"))?
                } else {
                    false
                };
                if dst < 8 {
                    return Err(error("scalar ALU would modify a resource descriptor"));
                }
                let (value, flag) = op.evaluate(src0, src1, scc_in);
                let slot = scalars
                    .get_mut(usize::from(dst))
                    .ok_or_else(|| error("scalar destination exceeds the fixture register file"))?;
                *slot = Some(value);
                if let Some(flag) = flag {
                    scc = Some(flag);
                }
                continue;
            }
            Instruction::ScalarCompare { op, src0, src1 } => {
                scc = Some(op.test(read_scalar(src0)?, read_scalar(src1)?));
                continue;
            }
            // docs/shaders/vector-compare.md sections 3 and 5.
            Instruction::VectorCompare { op, src0, src1 } => {
                if exec == 0 {
                    scalars[VCC_HI] = None;
                    scalars[VCC_LO] = Some(0);
                    lane_vcc = None;
                    continue;
                }
                let constant =
                    |operations: &mut Vec<(usize, Op)>, facts: &mut Vec<KnownBits>, bits: u32| {
                        let value = Value(operations.len());
                        operations.push((offset, Op::Constant(bits)));
                        facts.push(KnownBits::constant(bits));
                        value
                    };
                let src0 = match src0 {
                    VectorSource::Register(register) => read(register)?,
                    VectorSource::Scalar(register) if usize::from(register) == VCC_LO => {
                        match (lane_vcc, scalars[VCC_LO]) {
                            (Some(mask), _) => mask,
                            (None, Some(bits)) => constant(&mut operations, &mut facts, bits),
                            (None, None) => {
                                return Err(error("compare reads uninitialized VCC_LO"));
                            }
                        }
                    }
                    VectorSource::Scalar(register) => {
                        let bits = read_scalar(ScalarSource::Register(register))?;
                        constant(&mut operations, &mut facts, bits)
                    }
                    VectorSource::Immediate(bits) => constant(&mut operations, &mut facts, bits),
                };
                let src1 = read(src1)?;
                let mask = Value(operations.len());
                operations.push((
                    offset,
                    Op::LaneMask {
                        op,
                        src0,
                        src1,
                        exec,
                    },
                ));
                facts.push(KnownBits::UNKNOWN);
                scalars[VCC_LO] = None;
                scalars[VCC_HI] = None;
                lane_vcc = Some(mask);
                continue;
            }
            Instruction::BranchVccZero { target } if let Some(mask) = lane_vcc => {
                if pending_load.is_some() {
                    return Err(error(
                        "a runtime s_cbranch_vccz requires no outstanding vector load",
                    ));
                }
                region = Some((target, mask, registers));
                (None, Op::RuntimeBranch { mask, target })
            }
            Instruction::BranchSccZero { target }
            | Instruction::BranchExecZero { target }
            | Instruction::BranchVccZero { target } => {
                let taken = match instruction.instruction {
                    Instruction::BranchSccZero { .. } => {
                        !scc.ok_or_else(|| error("conditional branch reads uninitialized SCC"))?
                    }
                    Instruction::BranchVccZero { .. } => {
                        scalars[106]
                            .ok_or_else(|| error("conditional branch reads uninitialized VCC_LO"))?
                            == 0
                    }
                    _ => exec == 0,
                };
                if taken {
                    skip_before = target;
                }
                (None, Op::ScalarBranch { target, taken })
            }
            Instruction::Branch { target } => {
                skip_before = target;
                (
                    None,
                    Op::ScalarBranch {
                        target,
                        taken: true,
                    },
                )
            }
            Instruction::VectorAlu {
                op,
                dst,
                src0,
                src1,
            } => {
                let src0 = match src0 {
                    VectorSource::Register(register) => read(register)?,
                    VectorSource::Scalar(register) if usize::from(register) == VCC_LO => {
                        match (lane_vcc, scalars[VCC_LO]) {
                            (Some(mask), _) => mask,
                            (None, Some(bits)) => {
                                constant(&mut operations, &mut facts, offset, bits)
                            }
                            (None, None) => {
                                return Err(error("vector ALU reads uninitialized VCC_LO"));
                            }
                        }
                    }
                    VectorSource::Scalar(register) => {
                        let bits = read_scalar(ScalarSource::Register(register))?;
                        constant(&mut operations, &mut facts, offset, bits)
                    }
                    VectorSource::Immediate(bits) => {
                        constant(&mut operations, &mut facts, offset, bits)
                    }
                };
                let src1 = read(src1)?;
                let (left, right) = if op.swaps_sources() {
                    (src1, src0)
                } else {
                    (src0, src1)
                };
                (
                    Some(dst),
                    Op::Binary(Binary::Vector(op.lowering()), left, right),
                )
            }
            Instruction::VectorAddF32 { dst, src0, src1 } => (
                Some(dst),
                Op::Binary(Binary::FloatAdd, read(src0)?, read(src1)?),
            ),
            Instruction::VectorMulF32 { dst, src0, src1 } => (
                Some(dst),
                Op::Binary(Binary::FloatMul, read(src0)?, read(src1)?),
            ),
            Instruction::Export { .. }
            | Instruction::Interpolate { .. }
            | Instruction::ImageSampleLz { .. } => {
                return Err(error(
                    "graphics instruction is unsupported in a compute fixture",
                ));
            }
            Instruction::EndProgram => {
                if !stored || store_pending || pending_load.is_some() {
                    return Err(error(
                        "termination requires one completed store and no pending loads",
                    ));
                }
                return Ok(operations);
            }
        };
        let value_bits = match operation {
            Op::Constant(bits) => KnownBits::constant(bits),
            Op::Binary(Binary::Vector(VectorBinary::And), left, right) => {
                facts[left.0].and(facts[right.0])
            }
            Op::Binary(Binary::Vector(VectorBinary::Or), left, right) => {
                facts[left.0].or(facts[right.0])
            }
            Op::Binary(Binary::Vector(VectorBinary::Xor), left, right) => {
                facts[left.0].xor(facts[right.0])
            }
            Op::Binary(operator @ (Binary::FloatAdd | Binary::FloatMul), left, right) => {
                exact_float_result(operator, facts[left.0], facts[right.0])
                    .ok_or_else(|| error("FP32 operation is not proven exact for positive normal operands and results"))?
            }
            _ => KnownBits::UNKNOWN,
        };
        // Arms may push operand constants first, so the result is the next slot
        // now rather than `next`.
        let produced = Value(operations.len());
        facts.push(value_bits);
        operations.push((offset, operation));
        if let Some(dest) = dest {
            let slot = usize::from(dest);
            if dest == 0 || pending_load.is_some_and(|(register, _)| register == slot) {
                return Err(error(
                    "instruction writes the index register or replaces a pending load",
                ));
            }
            registers[slot] = Some(merge_lanes(
                &mut operations,
                &mut facts,
                offset,
                exec,
                produced,
                registers[slot],
            )?);
        }
    }
    Err(CompileError::Verify {
        offset: code_len,
        reason: "missing program termination",
    })
}

/// A wave-uniform constant operand at `offset`.
fn constant(
    operations: &mut Vec<(usize, Op)>,
    facts: &mut Vec<KnownBits>,
    offset: usize,
    bits: u32,
) -> Value {
    let value = Value(operations.len());
    operations.push((offset, Op::Constant(bits)));
    facts.push(KnownBits::constant(bits));
    value
}

// Keeping a value for inactive lanes requires a previously initialized destination.
fn merge_lanes(
    operations: &mut Vec<(usize, Op)>,
    facts: &mut Vec<KnownBits>,
    offset: usize,
    mask: u32,
    active: Value,
    inactive: Option<Value>,
) -> Result<Value, CompileError> {
    if mask == u32::MAX {
        return Ok(active);
    }
    let inactive = inactive.ok_or(CompileError::Verify {
        offset,
        reason: "partial EXEC requires an initialized vector destination",
    })?;
    let next = Value(operations.len());
    operations.push((
        offset,
        Op::MergeLanes {
            mask,
            active,
            inactive,
        },
    ));
    facts.push(KnownBits {
        zero: facts[active.0].zero & facts[inactive.0].zero,
        one: facts[active.0].one & facts[inactive.0].one,
        values: None,
    });
    Ok(next)
}

#[cfg(test)]
mod tests {
    use super::{Binary, KnownBits, exact_float_result};

    #[test]
    fn propagates_known_bits_through_integer_operators() {
        let left = KnownBits::constant(0b1100);
        let right = KnownBits::constant(0b1010);
        let and = left.and(right);
        assert_eq!((and.zero, and.one), (!0b1000, 0b1000));
        let xor = left.xor(right);
        assert_eq!((xor.zero, xor.one), (!0b0110, 0b0110));
    }

    #[test]
    fn rejects_unbounded_or_invalid_float_operands() {
        let one = KnownBits::constant(1.0f32.to_bits());
        let varying = KnownBits::UNKNOWN
            .and(KnownBits::constant(0x0040_0000))
            .xor(one);
        assert_eq!(varying.positive_normal_candidates(), Some([1.0, 1.5]));
        assert_eq!(one.and(one).positive_normal_candidates(), Some([1.0; 2]));
        assert!(KnownBits::UNKNOWN.positive_normal_candidates().is_none());
        for invalid in [0x8000_0000, 0xbf80_0000, 1, 0x7f80_0000] {
            assert!(
                KnownBits::constant(invalid)
                    .positive_normal_candidates()
                    .is_none()
            );
        }
    }

    #[test]
    fn keeps_only_bits_shared_by_every_exact_result() {
        let operand = KnownBits::UNKNOWN
            .and(KnownBits::constant(0x0040_0000))
            .xor(KnownBits::constant(1.0f32.to_bits()));
        let sum = exact_float_result(Binary::FloatAdd, operand, operand).unwrap();
        // The sums 2, 2.5 and 3 differ only in the two highest fraction bits.
        assert_eq!(sum.zero, 0xbf9f_ffff);
        assert_eq!(sum.one, 0x4000_0000);
        assert!(sum.values.is_none());
    }
}
