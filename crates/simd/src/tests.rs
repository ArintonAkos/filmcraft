use super::*;

/// 7.15.2 as the spec writes it: one sample at a time into the line sums, then the costs.
fn spec_direction(rows: &[&[u16; 8]; 8], shift: u32) -> (usize, i32) {
    const DIV: [i32; 9] = [0, 840, 420, 280, 210, 168, 140, 120, 105];
    let mut partial = [[0i32; 15]; 8];
    for i in 0..8 {
        for j in 0..8 {
            let x = (rows[i][j] >> shift) as i32 - 128;
            partial[0][i + j] += x;
            partial[1][i + j / 2] += x;
            partial[2][i] += x;
            partial[3][3 + i - j / 2] += x;
            partial[4][7 + i - j] += x;
            partial[5][3 - i / 2 + j] += x;
            partial[6][j] += x;
            partial[7][i / 2 + j] += x;
        }
    }
    let mut cost = [0i32; 8];
    for i in 0..8 {
        cost[2] += partial[2][i] * partial[2][i];
        cost[6] += partial[6][i] * partial[6][i];
    }
    cost[2] *= DIV[8];
    cost[6] *= DIV[8];
    for i in 0..7 {
        cost[0] += (partial[0][i] * partial[0][i] + partial[0][14 - i] * partial[0][14 - i]) * DIV[i + 1];
        cost[4] += (partial[4][i] * partial[4][i] + partial[4][14 - i] * partial[4][14 - i]) * DIV[i + 1];
    }
    cost[0] += partial[0][7] * partial[0][7] * DIV[8];
    cost[4] += partial[4][7] * partial[4][7] * DIV[8];
    for d in (1..8).step_by(2) {
        for j in 0..5 {
            cost[d] += partial[d][3 + j] * partial[d][3 + j];
        }
        cost[d] *= DIV[8];
        for j in 0..3 {
            cost[d] += (partial[d][j] * partial[d][j] + partial[d][10 - j] * partial[d][10 - j]) * DIV[2 * j + 2];
        }
    }
    cdef_pick(&cost)
}

/// Random blocks at 8, 10 and 12 bits, the extremes of each depth, and edges along every
/// direction.
fn blocks() -> Vec<([[u16; 8]; 8], u32)> {
    let mut seed = 0x2545_f491u32;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    let mut out = Vec::new();
    for shift in [0, 2, 4] {
        let max = (256u32 << shift) - 1;
        for _ in 0..10_000 {
            out.push((std::array::from_fn(|_| std::array::from_fn(|_| (next() % (max + 1)) as u16)), shift));
        }
        out.push(([[0; 8]; 8], shift));
        out.push(([[max as u16; 8]; 8], shift));
        for d in 0..8 {
            out.push((
                std::array::from_fn(|i| {
                    std::array::from_fn(|j| {
                        let (i, j) = (i as i32, j as i32);
                        let t = [i + j, 2 * i + j, i, 2 * i - j, i - j, j - 2 * i, j, j + 2 * i][d];
                        if t.rem_euclid(4) < 2 { max as u16 } else { 0 }
                    })
                }),
                shift,
            ));
        }
    }
    out
}

#[test]
fn every_level_matches_the_spec_direction_search() {
    for (block, shift) in blocks() {
        let rows: [&[u16; 8]; 8] = std::array::from_fn(|i| &block[i]);
        let want = spec_direction(&rows, shift);
        for &level in levels() {
            assert_eq!(cdef_direction_at(level, &rows, shift), want, "{level:?}, shift {shift}: {block:?}");
        }
    }
}

#[test]
fn native_level_follows_the_target() {
    let native = if cfg!(target_arch = "aarch64") {
        Level::Neon
    } else if cfg!(target_arch = "x86_64") {
        Level::Sse2
    } else {
        Level::Scalar
    };
    assert_eq!(Level::NATIVE, native);
    assert!(levels().contains(&Level::Scalar) && levels().contains(&native));
}

/// Filter strengths as 7.14.4 derives them from a level and a sharpness.
fn lf_thresh(lvl: i32, sharp: i32, bit_depth: u32) -> LfThresh {
    let shift = if sharp > 4 { 2 } else { (sharp > 0) as i32 };
    let limit = if sharp > 0 { (lvl >> shift).clamp(1, 9 - sharp) } else { (lvl >> shift).max(1) };
    LfThresh::new(limit, 2 * (lvl + 2) + limit, lvl >> 4, bit_depth)
}

/// 32x32 pictures around an edge through the middle: smooth sides (to reach the flat and wide
/// paths) or noisy ones, a step of random height or a random walk across the edge, random
/// strengths.
fn deblock_cases() -> Vec<(Vec<u16>, LfThresh, LfSize, u32, usize)> {
    let mut seed = 0x9e37_79b9u32;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    let mut out = Vec::new();
    for bd in [8, 10, 12] {
        let max = (1i32 << bd) - 1;
        let unit = 1i32 << (bd - 8);
        for size in [LfSize::Four, LfSize::Six, LfSize::Eight, LfSize::Sixteen] {
            for _ in 0..3000 {
                let noise = [0, 1, 2, 4, 16, 120][next() as usize % 6] * unit;
                let step = [0, 1, 2, 3, 8, 40, 100][next() as usize % 7] * unit;
                let level = (next() % (max as u32 + 1)) as i32;
                let vertical = next() % 2 == 0;
                // A third of the cases walk randomly across the edge (zigzags reach the narrow
                // filter's clamps); the rest are a step.
                let walk = next() % 3 == 0;
                let mut profile = [level; 32];
                for a in 1..32 {
                    let s = 70 * unit;
                    profile[a] =
                        if walk { (profile[a - 1] + (next() % (2 * s as u32 + 1)) as i32 - s).clamp(0, max) } else { level + if a >= 16 { step } else { 0 } };
                }
                // Lines differ in the size of the step, so lanes take different paths.
                let scale: [i32; 32] = std::array::from_fn(|_| [0, 1, 1, 2][next() as usize % 4]);
                let pic: Vec<u16> = (0..32 * 32)
                    .map(|i| {
                        let (across, line) = if vertical { (i % 32, i / 32) } else { (i / 32, i % 32) };
                        let n = if noise == 0 { 0 } else { (next() % (noise as u32 + 1)) as i32 - noise / 2 };
                        let p = if walk { profile[across] } else { level + if across >= 16 { step * scale[line] } else { 0 } };
                        (p + n).clamp(0, max) as u16
                    })
                    .collect();
                let t = lf_thresh(1 + (next() % 63) as i32, (next() % 8) as i32, bd);
                out.push((pic, t, size, vertical as u32, 4 * (1 + next() as usize % 4)));
            }
        }
    }
    out
}

#[test]
fn every_level_matches_the_scalar_loop_filter() {
    let (mut changed, mut total) = (0, 0);
    for (pic, t, size, vertical, lines) in deblock_cases() {
        let mut want = pic.clone();
        let base = if vertical == 1 { 8 * 32 + 16 } else { 16 * 32 + 8 };
        // Up to 16 lines from line 8: all inside the 32x32 picture.
        let run = |level: Level, d: &mut [u16]| {
            if vertical == 1 { deblock_vertical_at(level, d, base, 32, lines, size, &t) } else { deblock_horizontal_at(level, d, base, 32, lines, size, &t) }
        };
        run(Level::Scalar, &mut want);
        total += 1;
        changed += (want != pic) as u32;
        for &level in levels() {
            let mut got = pic.clone();
            run(level, &mut got);
            assert_eq!(got, want, "{level:?} {size:?} vertical {vertical} {lines} lines {t:?}");
        }
    }
    // The cases reach the filters, not only the masks that turn them off.
    assert!(changed > total / 3, "{changed} of {total}");
}

#[test]
fn loop_filter_falls_back_when_the_vector_rows_leave_the_buffer() {
    // A vertical edge 2 samples from the start: the 4-tap filter reads p1, but the vector
    // kernel's 8-sample rows would start before the buffer.
    let pic: Vec<u16> = (0..4 * 8).map(|i| if i % 8 >= 2 { 120 } else { 100 }).collect();
    let t = lf_thresh(40, 0, 8);
    let mut want = pic.clone();
    deblock_vertical_at(Level::Scalar, &mut want, 2, 8, 4, LfSize::Four, &t);
    assert_ne!(want, pic);
    for &level in levels() {
        let mut got = pic.clone();
        deblock_vertical_at(level, &mut got, 2, 8, 4, LfSize::Four, &t);
        assert_eq!(got, want, "{level:?}");
    }
}

#[test]
fn loop_filter_clamps_a_zigzag_at_full_strength() {
    // p1 - q1 is past the signed range the narrow filter clamps it to (7.14.6.3), while the
    // differences next to the edge stay within the strongest filter's limits.
    for bd in [8, 10, 12] {
        let u = 1 << (bd - 8);
        let line = [100, 100, 100, 100, 100, 100, 263, 200, 190, 127, 100, 100, 100, 100, 100, 100].map(|v: u16| v * u);
        let t = lf_thresh(63, 0, bd);
        for vertical in [false, true] {
            let pic: Vec<u16> = (0..16 * 16).map(|i| line[if vertical { i % 16 } else { i / 16 }]).collect();
            let base = if vertical { 4 * 16 + 8 } else { 8 * 16 + 4 }; // 8 lines from line 4
            let mut want = pic.clone();
            let run = |level, d: &mut [u16]| {
                if vertical {
                    deblock_vertical_at(level, d, base, 16, 8, LfSize::Four, &t)
                } else {
                    deblock_horizontal_at(level, d, base, 16, 8, LfSize::Four, &t)
                }
            };
            run(Level::Scalar, &mut want);
            assert_ne!(want, pic);
            for &level in levels() {
                let mut got = pic.clone();
                run(level, &mut got);
                assert_eq!(got, want, "{level:?} bd {bd} vertical {vertical}");
            }
        }
    }
}

#[test]
fn every_level_matches_the_scalar_subpel_filters() {
    let mut seed = 0x51ed_270bu32;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    // AV1's filter shapes: taps summing to 128 with small negative side lobes.
    for case in 0..20_000 {
        let bd = [8, 10, 12][case % 3];
        let max = (1u32 << bd) - 1;
        let mut taps = [0i16; 8];
        for t in &mut taps {
            *t = (next() % 25) as i16 - 12;
        }
        taps[3] = 128 - taps.iter().enumerate().filter(|&(i, _)| i != 3).map(|(_, &t)| t).sum::<i16>();
        let w = [1, 2, 3, 4, 5, 8, 12, 16, 20, 32, 64, 128][next() as usize % 12];
        let shift = if bd == 12 { 5 } else { 3 };
        let src: Vec<u16> = (0..w + 7).map(|_| if next() % 4 == 0 { max as u16 } else { (next() % (max + 1)) as u16 }).collect();
        let mut want = vec![0i16; w];
        subpel_h_at(Level::Scalar, &src, &taps, &mut want, shift);
        for &level in levels() {
            let mut got = vec![0i16; w];
            subpel_h_at(level, &src, &taps, &mut got, shift);
            assert_eq!(got, want, "{level:?} h w {w} bd {bd} taps {taps:?}");
        }
        let stride = w + (next() % 3) as usize;
        let rows: Vec<i16> = (0..7 * stride + w).map(|_| ((next() % 51_000) as i32 - 25_500) as i16).collect();
        let vshift = [7, 9, 11][next() as usize % 3];
        let mut want = vec![0i32; w];
        subpel_v_at(Level::Scalar, &rows, stride, &taps, &mut want, vshift);
        for &level in levels() {
            let mut got = vec![0i32; w];
            subpel_v_at(level, &rows, stride, &taps, &mut got, vshift);
            assert_eq!(got, want, "{level:?} v w {w} stride {stride} taps {taps:?}");
        }
    }
}

#[test]
fn every_level_matches_the_scalar_cdef_filter() {
    let mut seed = 0x0c0f_fee5u32;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    for case in 0..30_000 {
        let shift = [0, 2, 4][case % 3];
        let max = (256u32 << shift) - 1;
        let (w, h) = [(8, 8), (4, 4), (4, 8), (8, 4), (4, 2)][next() as usize % 5];
        let base = next() % (max + 1);
        let spread = [2, 16, 200, 8192][next() as usize % 4];
        let buf: [u16; CDEF_STRIDE * CDEF_STRIDE] = std::array::from_fn(|_| (base + next() % spread).saturating_sub(spread / 2).min(max) as u16);
        // AV1's taps: 2 rows and columns around the sample at most, weights up to 4.
        let n = 1 + next() as usize % 12;
        let taps: Vec<CdefTap> = (0..n)
            .map(|_| CdefTap {
                off: (next() % 5) as isize * CDEF_STRIDE as isize + (next() % 5) as isize - 2 * CDEF_STRIDE as isize - 2,
                weight: 1 + (next() % 4) as i16,
                strength: ((next() % 16) << shift) as i16,
                shift: next() % 11,
            })
            .collect();
        let clamp = next() % 2 == 0;
        let mut want = [0u16; 64];
        cdef_filter_at(Level::Scalar, &buf, w, h, &taps, clamp, &mut want);
        for &level in levels() {
            let mut got = [0u16; 64];
            cdef_filter_at(level, &buf, w, h, &taps, clamp, &mut got);
            assert_eq!(got, want, "{level:?} {w}x{h} clamp {clamp} taps {taps:?}");
        }
    }
}

#[test]
fn cdef_filter_falls_back_when_a_tap_leaves_the_buffer() {
    let buf: [u16; CDEF_STRIDE * CDEF_STRIDE] = std::array::from_fn(|i| (i * 7 % 50) as u16 + 100);
    // Three rows below the bottom row of an 8x8 block: past the end of the padded block.
    let taps = [CdefTap { off: 3 * CDEF_STRIDE as isize, weight: 4, strength: 12, shift: 1 }, CdefTap { off: -1, weight: 2, strength: 12, shift: 1 }];
    let mut want = [0u16; 64];
    cdef_filter_at(Level::Scalar, &buf, 8, 8, &taps, true, &mut want);
    for &level in levels() {
        let mut got = [0u16; 64];
        cdef_filter_at(level, &buf, 8, 8, &taps, true, &mut got);
        assert_eq!(got, want, "{level:?}");
    }
}
