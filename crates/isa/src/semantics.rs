//! Typed vocabulary of the declarative semantic registry (`crates/isa/semantics/rdna2.toml`).
//!
//! The registry is data. `crates/isa/generate_semantics.py` validates it against
//! this closed vocabulary and the pinned AMD XML snapshot, then emits
//! `semantics_gen.rs`: the typed operation enums, their opcode dispatch and one
//! static [`FormSemantics`] row per form. Nothing here interprets metadata at run
//! time; the decoder and translator call the generated `const fn`s and `match`es.
//!
//! The reference interpreter (`crates/ref`) and the independent interpreter
//! (`tests/validation/independent_ref.rs`) must not use this module.

use crate::CompareRelation;
use crate::meta::EncodingFamily;

pub use crate::semantics_gen::{FORM_IDS, FORMS, GENERATED_SCHEMA};

/// Version of the registry schema this crate was generated against.
pub const SCHEMA: u32 = 1;

/// Reusable semantic primitive selected by a form. Closed: the generator rejects
/// any other name, and each primitive has one implementation in `primitive.rs`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operation {
    /// Integer addition.
    Add,
    /// Integer subtraction.
    Sub,
    /// Bitwise AND.
    And,
    /// Bitwise OR.
    Or,
    /// Bitwise exclusive OR.
    Xor,
    /// Bitwise AND with the complement of S1.
    AndNot,
    /// Bitwise OR with the complement of S1.
    OrNot,
    /// Complement of bitwise AND.
    Nand,
    /// Complement of bitwise OR.
    Nor,
    /// Complement of bitwise exclusive OR.
    Xnor,
    /// Shift S0 left by the masked count in S1, filling with zeros.
    ShiftLeft,
    /// Shift S0 right by the masked count in S1, filling with zeros.
    LogicalShiftRight,
    /// Shift S0 right by the masked count in S1, filling with its sign bit.
    ArithmeticShiftRight,
    /// Compare S0 with S1 under the form's relation; writes only SCC.
    Compare,
    /// Bitwise complement of S0.
    Not,
    /// S0 with its bit order reversed.
    ReverseBits,
    /// Number of zero bits in S0.
    CountZeros,
    /// Number of one bits in S0.
    CountOnes,
    /// Index of the lowest zero bit of S0, or -1 when there is none.
    FindFirstZero,
    /// Index of the lowest one bit of S0, or -1 when there is none.
    FindFirstOne,
    /// Unsigned: count of zeros above the highest one bit; signed: count of bits
    /// above the highest bit that differs from the sign bit. -1 when none.
    FindLastBit,
    /// Sign-extend the low `source_bits` of S0 to 32 bits.
    SignExtend,
    /// Two's-complement absolute value; the most negative value maps to itself.
    Absolute,
    /// S0 when it is strictly less than S1, otherwise S1.
    Minimum,
    /// S0 when it is strictly greater than S1, otherwise S1.
    Maximum,
    /// S0 when SCC is set, otherwise S1.
    Select,
    /// `((1 << S0[4:0]) - 1) << S1[4:0]`.
    BitfieldMask,
    /// Low 32 bits of the product.
    Multiply,
    /// High 32 bits of the 64-bit product.
    MultiplyHigh,
    /// The field of S0 at offset S1[4:0] with width S1[22:16]; sign-extended when signed.
    BitfieldExtract,
    /// Absolute value of the wrapped 32-bit difference S0 - S1.
    AbsoluteDifference,
    /// `(S0 << shift_amount) + S1`.
    ShiftLeftAdd,
    /// Low 32 bits of the product of S0[23:0] and S1[23:0] (sign-extended when signed).
    Multiply24,
    /// Bits 63:32 of the product of S0[23:0] and S1[23:0] (sign-extended when signed).
    MultiplyHigh24,
}

/// The per-lane integer operation a vector form lowers to. One reusable MSL
/// template exists per variant; a form selects it through its declared
/// operation and signedness, never through its opcode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VectorBinary {
    /// Wrapping `a + b`.
    Add,
    /// Wrapping `a - b`.
    Sub,
    /// `a & b`.
    And,
    /// `a | b`.
    Or,
    /// `a ^ b`.
    Xor,
    /// `!(a ^ b)`.
    Xnor,
    /// `a << b[4:0]`. Reversed forms pass the count as `b`, so `a` is S1.
    ShiftLeft,
    /// `a >> b[4:0]`, filling with zeros.
    LogicalShiftRight,
    /// `a >> b[4:0]`, filling with the sign bit of `a`.
    ArithmeticShiftRight,
    /// Two's-complement minimum.
    MinSigned,
    /// Unsigned minimum.
    MinUnsigned,
    /// Two's-complement maximum.
    MaxSigned,
    /// Unsigned maximum.
    MaxUnsigned,
    /// Low 32 bits of `sext(a[23:0]) * sext(b[23:0])`.
    Mul24Signed,
    /// Low 32 bits of `a[23:0] * b[23:0]`.
    Mul24Unsigned,
    /// Bits 63:32 of `sext(a[23:0]) * sext(b[23:0])`.
    MulHi24Signed,
    /// Bits 63:32 of `a[23:0] * b[23:0]`.
    MulHi24Unsigned,
}

/// How the operands are interpreted. `Bitwise` exactly when the operation is logical.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Signedness {
    /// Unsigned integers.
    Unsigned,
    /// Two's-complement integers.
    Signed,
    /// Untyped bits.
    Bitwise,
}

/// Whether the operation reads SCC.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CarryIn {
    /// SCC is not read.
    None,
    /// SCC as carry-in (add), borrow-in (sub) or the select condition (select).
    Scc,
}

/// What a form writes to SCC.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SccEffect {
    /// SCC is not written.
    Unchanged,
    /// Unsigned carry-out of the top bit.
    CarryOut,
    /// Unsigned borrow (subtrahend plus borrow-in exceeds the minuend).
    Borrow,
    /// Two's-complement overflow of the exact result.
    SignedOverflow,
    /// The result is not zero.
    Nonzero,
    /// The compare's relation holds for S0 and S1.
    Relation,
    /// The result is S0 by the strict comparison (min, max): equal operands give 0.
    SelectedSrc0,
}

/// What a form writes to VCC or EXEC. Stage A forms leave both unchanged.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaskEffect {
    /// Not written.
    Unchanged,
}

/// Wave-size restriction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Wave {
    /// Behaviour does not depend on the wave size.
    Any,
    /// Only Wave32 behaviour is declared; Wave64 stays unsupported.
    Wave32,
}

/// Shared operand rule. The decoder implements each rule once.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OperandRule {
    /// SOP2: SDST is s0-s105; SSRC0/SSRC1 are s0-s105, inline integers -16..64
    /// or one literal word. Everything else is rejected as unsupported.
    Sop2Sgpr,
    /// SOPC: no destination; SSRC0/SSRC1 as for SOP2 sources (s0-s105, inline
    /// integers -16..64 or one literal word). Everything else is rejected.
    SopcScalar,
    /// SOP1 unary: SDST is s0-s105; SSRC0 is s0-s105, an inline integer -16..64 or
    /// a literal word. Everything else is rejected as unsupported.
    Sop1Sgpr,
    /// VOP2 vector: VDST is v0-v255; SRC0 is a VGPR, s0-s105, `vcc_lo`, an inline
    /// integer -16..64 or a literal word; VSRC1 is a VGPR. Everything else is
    /// rejected as unsupported.
    Vop2Vgpr,
}

/// Evidence a form needs before it counts as verified.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Evidence {
    /// `crates/ref` lockstep reference.
    Reference,
    /// `tests/validation/independent_ref.rs`.
    Independent,
    /// Native Metal known-answer runs.
    Metal,
    /// LLVM assembler/disassembler encoding agreement.
    LlvmEncoding,
}

/// Public document a [`SpecRef`] points at.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Document {
    /// AMD RDNA 2 ISA reference guide (PDF).
    Guide,
    /// Pinned AMD machine-readable XML.
    Xml,
}

/// One public specification reference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpecRef {
    /// Source document.
    pub document: Document,
    /// Section number, for example `"12.1"`.
    pub section: &'static str,
    /// Page in the document, or 0 when the page has not been recorded.
    pub page: u16,
}

/// One validated registry row, emitted as a static table by the generator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FormSemantics {
    /// XML instruction name, the stable form identity.
    pub name: &'static str,
    /// Encoding family (matches [`crate::meta::FORMS`]).
    pub family: EncodingFamily,
    /// Opcode within the family.
    pub opcode: u16,
    /// Primitive.
    pub operation: Operation,
    /// Operand interpretation.
    pub signedness: Signedness,
    /// SCC input.
    pub carry_in: CarryIn,
    /// Operand and result width in bits.
    pub width: u8,
    /// Low bits of S1 used as the shift count by shift operations; 0 otherwise.
    pub shift_count_bits: u8,
    /// Fixed left shift of S0 for shift-and-add operations (1 to 4); 0 otherwise.
    pub shift_amount: u8,
    /// Whether the operation reads its sources in reverse order (`D = op(S1, S0)`).
    pub swap_sources: bool,
    /// Low bits of S0 that the operation reads: `width`, or 8 or 16 for sign extension.
    pub source_bits: u8,
    /// The relation tested by compares (S0 on the left); `None` otherwise.
    pub relation: Option<CompareRelation>,
    /// SCC write.
    pub scc: SccEffect,
    /// VCC write.
    pub vcc: MaskEffect,
    /// EXEC write.
    pub exec: MaskEffect,
    /// Wave-size restriction.
    pub wave: Wave,
    /// Operand rule.
    pub operands: OperandRule,
    /// Public references.
    pub spec: &'static [SpecRef],
    /// Required evidence.
    pub verify: &'static [Evidence],
}
