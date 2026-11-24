use xenora_metal::{DepthFill, IndexBuffer, IndexedDraw, IndexedPass};

use super::indexed::{bc1_upload, metal_draws};
use super::indexed_cases::{Fill, INDICES, SIZE, cases, expected, vertices, wide_indices};
use super::{context, native};

/// Every indexed case drawn through the mesh pipeline must equal the same
/// independently derived image the indexed draw is checked against, per pixel.
/// The mesh function fetches the indices and vertices itself, one mesh
/// threadgroup per triangle, so no expected value comes from the mesh path.
#[test]
fn mesh_draw_known_answers() {
    let gpu = context();
    let upload = bc1_upload();
    let vertices = vertices();
    let wide = wide_indices();
    let (cases, expected) = (cases(), expected());
    assert_eq!(cases.len(), expected.len());
    let mut checked = 0;
    for (case, want) in cases.iter().zip(&expected) {
        assert_eq!(case.name, want.name);
        let draws = metal_draws(case);
        let color = native(
            gpu.render_mesh(&IndexedPass {
                extent: [SIZE, SIZE],
                clear_color: case.clear_color,
                vertices: &vertices,
                indices: if case.wide {
                    IndexBuffer::U32(&wide)
                } else {
                    IndexBuffer::U16(&INDICES)
                },
                draws: &draws,
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
            assert_eq!(pixel, wanted, "{} mesh colour at ({x}, {y})", case.name);
        }
        checked += 1;
    }
    println!(
        "checked {checked} mesh-pipeline passes of {} pixels",
        SIZE * SIZE
    );
}

/// The mesh pass shares the indexed pass's validation, which runs before the
/// Metal 3 capability check, so malformed draws are rejected on every device.
#[test]
fn invalid_mesh_draws_are_rejected() {
    let gpu = context();
    let vertices = vertices();
    let flat = DepthFill::Flat([255; 4]);
    let run = |indices, start_index, index_count, base_vertex, fill| {
        gpu.render_mesh(&IndexedPass {
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
    for (reason, error, wanted) in [
        (
            "count not a multiple of 3",
            run(short, 0, 4, 0, flat),
            "index count must be a nonzero multiple of 3",
        ),
        (
            "range past buffer",
            run(short, 19, 3, 0, flat),
            "index range past the index buffer",
        ),
        (
            "negative vertex",
            run(short, 0, 3, -1, flat),
            "outside the vertex buffer",
        ),
        (
            "16-bit restart value",
            run(IndexBuffer::U16(&[0, 1, u16::MAX]), 0, 3, 0, flat),
            "primitive-restart value",
        ),
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
