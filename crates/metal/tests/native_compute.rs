use xenora_metal::NativeCompute;

#[path = "native/blend.rs"]
mod blend;
#[path = "support/blend_cases.rs"]
mod blend_cases;
#[path = "native/depth.rs"]
mod depth;
#[path = "support/depth_cases.rs"]
mod depth_cases;
#[path = "native/guest_buffer.rs"]
mod guest_buffer;
#[path = "native/indexed.rs"]
mod indexed;
#[path = "support/indexed_cases.rs"]
mod indexed_cases;
#[path = "native/mesh.rs"]
mod mesh;
#[path = "native/offscreen.rs"]
mod offscreen;
#[path = "native/texture.rs"]
mod texture;
#[path = "native/texture_bc_alpha.rs"]
mod texture_bc_alpha;
#[path = "native/texture_bc_channel.rs"]
mod texture_bc_channel;
#[path = "native/texture_sampling.rs"]
mod texture_sampling;
#[path = "native/translated_render.rs"]
mod translated_render;

const KERNEL_HEADER: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void fixture(device const uint *input [[buffer(0)]],
                    device uint *output [[buffer(1)]],
                    uint id [[thread_position_in_grid]],
                    uint local [[thread_index_in_threadgroup]],
                    uint group_size [[threads_per_threadgroup]]) {
";

fn fixture(body: &str) -> String {
    format!("{KERNEL_HEADER}{body}\n}}")
}

fn context() -> NativeCompute {
    NativeCompute::new().expect("native Metal device and queue")
}

/// Unwraps a native result, reporting a missing device capability apart from
/// an implementation failure. Both fail the test; neither counts as a pass.
fn native<T>(result: Result<T, xenora_metal::MetalError>) -> T {
    result.unwrap_or_else(|error| {
        if error.is_capability_unavailable() {
            panic!("CAPABILITY UNAVAILABLE on this device, not an implementation failure: {error}")
        }
        panic!("IMPLEMENTATION FAILURE: {error}")
    })
}

fn verify_words(body: &str, input: &[u32], expected: &[u32]) {
    let actual = context()
        .run_words(&fixture(body), "fixture", input)
        .expect("compiled and completed native Metal dispatch");
    assert_eq!(actual, expected);
}

const MAX_DISPATCH_GROUPS: usize = 64;

fn mixing_fixture(rounds: u32) -> String {
    fixture(&format!(
        "
        uint value = input[id] ^ (id * 0x9e3779b9u);
        for (uint round = 0u; round < {rounds}u; ++round) {{
            value ^= value << 13u;
            value ^= value >> 17u;
            value ^= value << 5u;
            value = value * 0x85ebca6bu + round;
        }}
        output[id] = value;
        "
    ))
}

fn mixing_expectation(input: &[u32], rounds: u32) -> Vec<u32> {
    (0_u32..)
        .zip(input)
        .map(|(id, &word)| {
            let mut value = word ^ id.wrapping_mul(0x9e37_79b9);
            for round in 0..rounds {
                let left = value ^ (value << 13);
                let right = left ^ (left >> 17);
                let mixed = right ^ (right << 5);
                value = mixed.wrapping_mul(0x85eb_ca6b).wrapping_add(round);
            }
            value
        })
        .collect()
}

fn mixing_input(words: u32) -> Vec<u32> {
    (0..words)
        .map(|id| id.wrapping_mul(0xc2b2_ae35) ^ 0xdead_beef)
        .collect()
}

#[test]
fn threadgroup_widths_give_identical_known_answers() {
    const WIDTHS: [usize; 16] = [
        1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 96, 128, 255, 256, 512, 1024,
    ];
    const GRIDS: [u32; 7] = [1, 63, 64, 65, 1000, 4093, 4096];
    const ROUNDS: u32 = 37;
    let context = context();
    let source = mixing_fixture(ROUNDS);
    let limit = native(context.max_group_width(&source, "fixture"));
    let mut checked = 0;
    let mut skipped = 0;

    for grid in GRIDS {
        let input = mixing_input(grid);
        let expected = mixing_expectation(&input, ROUNDS);
        for width in WIDTHS {
            if width > limit {
                println!("skipped grid={grid} width={width}: device/pipeline limit is {limit}");
                skipped += 1;
                continue;
            }
            let groups = input.len().div_ceil(width);
            if groups > MAX_DISPATCH_GROUPS {
                println!(
                    "skipped grid={grid} width={width}: {groups} groups exceed {MAX_DISPATCH_GROUPS}"
                );
                skipped += 1;
                continue;
            }
            let profile =
                native(context.run_words_with_group_width(&source, "fixture", &input, width));
            assert_eq!(profile.output, expected, "grid={grid} width={width}");
            checked += 1;
        }
    }
    assert!(checked > 0);
    println!("checked {checked} configurations, skipped {skipped}");
}

#[test]
fn threadgroup_width_rejections() {
    let context = context();
    let source = fixture("output[id] = input[id];");
    let limit = native(context.max_group_width(&source, "fixture"));
    for width in [0, limit + 1] {
        let error = context
            .run_words_with_group_width(&source, "fixture", &[25], width)
            .expect_err("invalid threadgroup width must fail");
        assert_eq!(error.operation, "threadgroup width");
    }
    let error = context
        .run_words_with_group_width(&source, "fixture", &[25; MAX_DISPATCH_GROUPS + 1], 1)
        .expect_err("too many threadgroups must fail");
    assert_eq!(error.operation, "dispatch bounds");
}

fn timing_summary(samples: &mut [f64]) -> String {
    samples.sort_by(f64::total_cmp);
    let count = samples.len();
    let median = f64::midpoint(samples[(count - 1) / 2], samples[count / 2]);
    let p95 = samples[(count * 95).div_ceil(100) - 1];
    format!(
        "median={:.3} min={:.3} max={:.3} p95={:.3}",
        median * 1e6,
        samples[0] * 1e6,
        samples[count - 1] * 1e6,
        p95 * 1e6,
    )
}

#[test]
#[ignore = "manual GPU timing run"]
fn threadgroup_width_benchmark() {
    const ROUNDS: u32 = 2000;
    const WARMUPS: usize = 5;
    const SAMPLES: usize = 30;
    let context = context();
    let device = context.device_name();
    let source = mixing_fixture(ROUNDS);
    let input = mixing_input(4096);
    let expected = mixing_expectation(&input, ROUNDS);
    let limit = native(context.max_group_width(&source, "fixture"));

    let mut widths = vec![None];
    for width in [32, 64, 128, 256, 512, 1024] {
        if width > limit {
            println!("device={device} skipped width={width}: device/pipeline limit is {limit}");
        } else if input.len().div_ceil(width) > MAX_DISPATCH_GROUPS {
            println!("device={device} skipped width={width}: exceeds {MAX_DISPATCH_GROUPS} groups");
        } else {
            widths.push(Some(width));
        }
    }
    let mut timings = vec![([0.0; SAMPLES], [0.0; SAMPLES]); widths.len()];
    for round in 0..WARMUPS + SAMPLES {
        for (width, (gpu, host)) in widths.iter().zip(&mut timings) {
            let profile = native(match width {
                Some(width) => {
                    context.run_words_with_group_width(&source, "fixture", &input, *width)
                }
                None => context.run_words_profiled(&source, "fixture", &input),
            });
            assert_eq!(profile.output, expected, "width={width:?} round={round}");
            if round >= WARMUPS {
                gpu[round - WARMUPS] = profile.gpu_seconds;
                host[round - WARMUPS] = profile.host_seconds;
            }
        }
    }
    for (width, (gpu, host)) in widths.into_iter().zip(&mut timings) {
        let width = width.map_or_else(|| "default".to_owned(), |width| width.to_string());
        println!(
            "device={device} width={width} gpu_us[{}] host_us[{}]",
            timing_summary(gpu),
            timing_summary(host),
        );
    }
}

#[test]
fn integer_multiplication() {
    let context = context();
    println!(
        "device={} api_validation_requested={} shader_validation_requested={}",
        context.device_name(),
        std::env::var("MTL_DEBUG_LAYER").as_deref() == Ok("1"),
        std::env::var("MTL_SHADER_VALIDATION").as_deref() == Ok("1")
    );

    let values = [0, 1, 2, 3, u32::MAX, 0x8000_0000, 0x4000_0000];
    let input: Vec<_> = values.into_iter().cycle().take(131).collect();
    let expected: Vec<_> = input.iter().map(|&value| value.wrapping_mul(7)).collect();
    let actual = context
        .run_words(&fixture("output[id] = input[id] * 7u;"), "fixture", &input)
        .expect("compiled and completed native Metal dispatch");
    assert_eq!(actual, expected);
    println!(
        "checked {} output words in a nonuniform grid",
        expected.len()
    );
}

#[test]
fn vector_addition() {
    verify_words(
        "output[id] = input[id] + input[id ^ 1u];",
        &[u32::MAX, 2, 0x8000_0000, 0x8000_0001, 123, 42, 0, 0],
        &[1, 1, 1, 1, 165, 165, 0, 0],
    );
}

#[test]
fn bitwise_operations() {
    let input = [0, 1, u32::MAX, 0x8000_0000, 0x1122_3344, 0x5555_AAAA];
    let expected = input.map(|value| ((value & 0x0F0F_0000) >> 8) ^ (!value & 0xFF));
    verify_words(
        "output[id] = ((input[id] & 0x0f0f0000u) >> 8u) ^ (~input[id] & 0xffu);",
        &input,
        &expected,
    );
}

#[test]
fn copy_across_distinct_bindings() {
    let input = [0x0123_4567, 0, u32::MAX, 0x8000_0000, 0x89AB_CDEF];
    verify_words("output[id] = input[id];", &input, &input);
}

#[test]
fn exactly_representable_floating_point_arithmetic() {
    let input = [0.0_f32, 0.5, -0.5, 1.25, -2.0, 1024.0];
    let expected = [0.25_f32, 1.25, -0.75, 2.75, -3.75, 2048.25];
    verify_words(
        "output[id] = as_type<uint>(as_type<float>(input[id]) * 2.0f + 0.25f);",
        &input.map(f32::to_bits),
        &expected.map(f32::to_bits),
    );
}

#[test]
fn conditional_branching() {
    verify_words(
        r"
        if (input[id] > 10u) {
            output[id] = input[id] - 10u;
        } else {
            output[id] = input[id] * 4u;
        }
        ",
        &[0, 10, 11, 3, u32::MAX, 17],
        &[0, 40, 1, 12, u32::MAX - 10, 7],
    );
}

#[test]
fn synchronized_threadgroup_memory() {
    let input: Vec<_> = (100..228).collect();
    let expected: Vec<_> = (0..input.len()).map(|index| input[index ^ 1]).collect();
    verify_words(
        r"
        threadgroup uint stored[64];
        stored[local] = input[id];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint neighbor = local ^ 1u;
        output[id] = neighbor < group_size ? stored[neighbor] : 0u;
        ",
        &input,
        &expected,
    );
}

#[test]
fn threadgroup_atomic_addition() {
    verify_words(
        r"
        threadgroup atomic_uint sum;
        if (local == 0u) {
            atomic_store_explicit(&sum, 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        atomic_fetch_add_explicit(&sum, input[id], memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        output[id] = atomic_load_explicit(&sum, memory_order_relaxed);
        ",
        &[5, 0, 2, 1, 17, 4, 3, 8],
        &[40; 8],
    );
}

#[test]
fn simdgroup_reduction() {
    let input: Vec<_> = (0..128).map(|index| 1 + (index & 1)).collect();
    // The historical M5 probe measured 32 lanes. Every pair sums to 3,
    // so a complete 32-lane SIMD group of this input sums to 48.
    verify_words("output[id] = simd_sum(input[id]);", &input, &[48; 128]);
}

#[test]
fn multiple_submissions() {
    let context = context();
    let source = fixture("output[id] = input[id] + id;");
    let first = context.run_words(&source, "fixture", &[1, 2, 3]).unwrap();
    let second = context
        .run_words(&source, "fixture", &[5, 4, u32::MAX, 2])
        .unwrap();
    assert_eq!(first, [1, 3, 5]);
    assert_eq!(second, [5, 5, 1, 5]);
}

#[test]
fn errors_propagate() {
    let context = context();
    let source = fixture("output[id] = input[id];");
    let error = context
        .run_words("not valid MSL", "fixture", &[25])
        .expect_err("invalid source must fail");
    assert_eq!(error.operation, "newLibraryWithSource");
    assert!(!error.detail.is_empty());
    let error = context
        .run_words(&source, "nonexistent_entry", &[25])
        .expect_err("missing entry must fail");
    assert_eq!(error.operation, "newFunctionWithName");
    let error = context
        .run_words(&source, "fixture", &[])
        .expect_err("empty dispatch must fail");
    assert_eq!(error.operation, "dispatch bounds");
    let error = context
        .run_words(&source, "fixture", &[0; 4097])
        .expect_err("excessive dispatch must fail");
    assert_eq!(error.operation, "dispatch bounds");
}

/// Identical outputs alone cannot show which width ran, so each thread also
/// reports the requested group width, its own group's size and its in-group
/// index (11, 11 and 10 bits).
#[test]
fn requested_threadgroup_width_is_dispatched() {
    const SOURCE: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void fixture(device const uint *input [[buffer(0)]],
                    device uint *output [[buffer(1)]],
                    uint id [[thread_position_in_grid]],
                    uint local [[thread_index_in_threadgroup]],
                    uint group_size [[threads_per_threadgroup]],
                    uint requested [[dispatch_threads_per_threadgroup]]) {
    output[id] = (requested << 21u) | (group_size << 10u) | local;
}
";
    let context = context();
    let limit = native(context.max_group_width(SOURCE, "fixture"));
    for (grid, width) in [(1000_usize, 33_usize), (4093, 64), (4096, 256), (65, 1024)] {
        if width > limit {
            println!("skipped grid={grid} width={width}: device/pipeline limit is {limit}");
            continue;
        }
        let profile =
            native(context.run_words_with_group_width(SOURCE, "fixture", &vec![0; grid], width));
        for (id, word) in profile.output.iter().enumerate() {
            // dispatchThreads trims the last group to the grid's remainder.
            let size = width.min(grid - id / width * width);
            let expected = u32::try_from((width << 21) | (size << 10) | (id % width)).unwrap();
            assert_eq!(*word, expected, "grid={grid} width={width} id={id}");
        }
    }
}
