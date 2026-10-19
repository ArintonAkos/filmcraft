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
