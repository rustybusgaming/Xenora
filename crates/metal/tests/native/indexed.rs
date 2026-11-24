use xenora_metal::{DepthFill, IndexBuffer, IndexedDraw, IndexedPass};
use xenora_texture::{Bc1Upload, Bc1UploadDesc};

use super::indexed_cases::{
    BC1_BLOCKS, Case, Fill, INDICES, SIZE, cases, expected, vertices, wide_indices,
};
use super::{context, native};

pub(super) fn metal_draws(case: &Case) -> Vec<IndexedDraw> {
    case.draws
        .iter()
        .map(|draw| IndexedDraw {
            start_index: draw.start,
            index_count: draw.count,
            base_vertex: draw.base_vertex,
            fill: match draw.fill {
                Fill::Flat(rgba) => DepthFill::Flat(rgba),
                Fill::Texture => DepthFill::Bc1Texture,
            },
        })
        .collect()
}

pub(super) fn bc1_upload() -> Bc1Upload<'static> {
    Bc1Upload::new(
        &BC1_BLOCKS,
        Bc1UploadDesc {
            width: 8,
            height: 8,
            mips: 1,
            layers: 1,
            format: 1,
            usage: 1,
            row_pitch: 16,
            slice_pitch: 32,
        },
    )
    .expect("valid BC1 upload")
}

/// Each case's native colour must equal the independently derived image
/// (which the CPU reference also reproduces in `indexed_reference`), per pixel.
/// Metal fetches the vertices itself from the index buffer, start offset and
/// base vertex of each `drawIndexedPrimitives` call.
#[test]
fn indexed_draw_known_answers() {
    let gpu = context();
    let upload = bc1_upload();
    let vertices = vertices();
    let wide = wide_indices();
    let (cases, expected) = (cases(), expected());
    assert_eq!(cases.len(), expected.len());
    let mut checked = 0;
    for (case, want) in cases.iter().zip(&expected) {
        assert_eq!(case.name, want.name);
        // One independently derived fetch list per draw keeps the fixtures aligned.
        assert_eq!(want.fetch.len(), case.draws.len(), "{}", case.name);
        let draws = metal_draws(case);
        let color = native(
            gpu.render_indexed(&IndexedPass {
                extent: [SIZE, SIZE],
                clear_color: case.clear_color,
                vertices: &vertices,
                indices: if case.wide {
                    IndexBuffer::U32(&wide)
                } else {
                    IndexBuffer::U16(&INDICES)
                },
                draws: &draws,
                // Only textured cases need BC support, so a device without it fails
                // those cases alone, as a capability error.
                texture: case
                    .draws
                    .iter()
                    .any(|draw| draw.fill == Fill::Texture)
                    .then_some(&upload),
            }),
        );
        assert_eq!(color.len(), SIZE * SIZE * 4);
        for (index, (pixel, wanted)) in color.as_chunks::<4>().0.iter().zip(&want.color).enumerate()
        {
            let (x, y) = (index % SIZE, index / SIZE);
            assert_eq!(pixel, wanted, "{} colour at ({x}, {y})", case.name);
        }
        checked += 1;
    }
    println!("checked {checked} indexed passes of {} pixels", SIZE * SIZE);
}

/// Every malformed draw is rejected before encoding, with its own reason.
#[test]
fn invalid_indexed_draws_are_rejected() {
    let gpu = context();
    let vertices = vertices();
    let restart = [0, 1, u16::MAX];
    let flat = DepthFill::Flat([255; 4]);
    let run = |indices, start_index, index_count, base_vertex, fill| {
        gpu.render_indexed(&IndexedPass {
            extent: [SIZE, SIZE],
            clear_color: [0; 4],
            vertices: &vertices,
            indices,
            draws: &[IndexedDraw {
                start_index,
                index_count,
                base_vertex,
                fill,
            }],
            texture: None,
        })
        .expect_err("invalid draw must be rejected")
        .to_string()
    };
    let short = IndexBuffer::U16(&INDICES);
    let count = "index count must be a nonzero multiple of 3";
    let range = "index range past the index buffer";
    let outside = "outside the vertex buffer";
    let restart_value = "primitive-restart value";
    for (reason, error, wanted) in [
        // First: Metal itself would draw this one, so only the check rejects it.
        (
            "count not a multiple of 3",
            run(short, 0, 4, 0, flat),
            count,
        ),
        ("zero count", run(short, 0, 0, 0, flat), count),
        ("range past buffer", run(short, 19, 3, 0, flat), range),
        ("start overflow", run(short, usize::MAX, 3, 0, flat), range),
        ("negative vertex", run(short, 0, 3, -1, flat), outside),
        ("vertex past buffer", run(short, 9, 3, 2, flat), outside),
        (
            "32-bit index past buffer",
            run(IndexBuffer::U32(&[0, 1, 13]), 0, 3, 0, flat),
            outside,
        ),
        (
            "16-bit restart value",
            run(IndexBuffer::U16(&restart), 0, 3, 0, flat),
            restart_value,
        ),
        // In range after the base vertex, but still the raw restart value.
        (
            "16-bit restart value with base vertex",
            run(IndexBuffer::U16(&[65533, 65534, 65535]), 0, 3, -65533, flat),
            restart_value,
        ),
        (
            "32-bit restart value",
            run(IndexBuffer::U32(&[0, 1, u32::MAX]), 0, 3, 0, flat),
            restart_value,
        ),
        ("base overflow", run(short, 0, 3, i32::MAX, flat), outside),
        (
            "texture missing",
            run(short, 0, 3, 0, DepthFill::Bc1Texture),
            "textured draw without texture",
        ),
    ] {
        println!("{reason}: {error}");
        assert!(
            error.contains("indexed pass") && error.contains(wanted),
            "{reason}: {error}"
        );
    }
}
