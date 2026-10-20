//! Hand-written SIMD kernels for the codecs' hot loops.
//!
//! Each kernel has a scalar version and one per architecture, chosen when the crate is built:
//!
//! - `aarch64` (Apple silicon, Windows and Linux on ARM): NEON, which every ARMv8-A core has;
//! - `x86_64`: SSE2, which every x86-64 core has;
//! - anything else (wasm32 among them): the scalar version.
//!
//! No path needs a runtime CPU check, so there is no dispatch cost and no code that only some
//! machines run. Every version gives the same bits as the scalar one; the tests check each
//! available version against it, and `FILMCRAFT_SIMD=scalar` forces the scalar one at run time
//! (to compare speed, or to rule the vector code out when chasing a bug).
//!
//! `unsafe` is confined to the architecture modules: entering a kernel compiled for the target's
//! baseline vector set, and vector loads and stores of fixed-size arrays. Kernels that are long
//! enough to share between architectures (the loop filter) are written once as a macro over a
//! handful of vector operations each module defines.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

mod deblock;
mod scalar;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon;

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod sse2;

use std::sync::OnceLock;

pub use deblock::{LfSize, LfThresh};

/// The instruction set a kernel runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Scalar,
    Neon,
    Sse2,
}

impl Level {
    /// The vector instruction set of this build's target, if the crate has kernels for it.
    pub const NATIVE: Level = if cfg!(target_arch = "aarch64") {
        Level::Neon
    } else if cfg!(target_arch = "x86_64") {
        Level::Sse2
    } else {
        Level::Scalar
    };
}

/// The level the kernels use: [`Level::NATIVE`], or scalar when `FILMCRAFT_SIMD=scalar`.
pub fn level() -> Level {
    static LEVEL: OnceLock<Level> = OnceLock::new();
    *LEVEL.get_or_init(|| if std::env::var("FILMCRAFT_SIMD").is_ok_and(|v| v.eq_ignore_ascii_case("scalar")) { Level::Scalar } else { Level::NATIVE })
}

/// The levels this build can run: scalar, and the native one when it differs.
pub fn levels() -> &'static [Level] {
    if Level::NATIVE == Level::Scalar { &[Level::Scalar] } else { &[Level::Scalar, Level::NATIVE] }
}

/// AV1 CDEF direction search (7.15.2) of an 8x8 block of samples of a `bit_depth - 8` = `shift`
/// bit deeper picture: the direction (0..8) and the variance term `(best - opposite) >> 10`.
pub fn cdef_direction(rows: &[&[u16; 8]; 8], shift: u32) -> (usize, i32) {
    cdef_direction_at(level(), rows, shift)
}

/// [`cdef_direction`] on a given level; a level this build has no kernels for runs scalar.
pub fn cdef_direction_at(level: Level, rows: &[&[u16; 8]; 8], shift: u32) -> (usize, i32) {
    match level {
        #[cfg(target_arch = "aarch64")]
        Level::Neon => neon::cdef_direction(rows, shift),
        #[cfg(target_arch = "x86_64")]
        Level::Sse2 => sse2::cdef_direction(rows, shift),
        _ => scalar::cdef_direction(rows, shift),
    }
}

/// AV1 loop filter (7.14.6) of one line across an edge: `d[base]` is q0 and `step` the distance
/// between samples across the edge. Scalar on every level (it filters lines the vector kernels
/// can't reach, such as those copied out across a band edge).
pub fn deblock_line(d: &mut [u16], base: usize, step: usize, size: LfSize, t: &LfThresh) {
    scalar::deblock_line(d, base, step, size, t);
}

/// AV1 loop filter (7.14.6) of `lines` (a multiple of 4) lines across a vertical edge with the
/// same filter: `d[base]` is q0 of the first line, and each next line starts `stride` further.
pub fn deblock_vertical(d: &mut [u16], base: usize, stride: usize, lines: usize, size: LfSize, t: &LfThresh) {
    deblock_vertical_at(level(), d, base, stride, lines, size, t);
}

/// [`deblock_vertical`] on a given level.
pub fn deblock_vertical_at(level: Level, d: &mut [u16], base: usize, stride: usize, lines: usize, size: LfSize, t: &LfThresh) {
    deblock_lines(lines, |first, n| {
        let at = base + first * stride;
        let done = match level {
            #[cfg(target_arch = "aarch64")]
            Level::Neon => neon::deblock_vertical(d, at, stride, n, size, t),
            #[cfg(target_arch = "x86_64")]
            Level::Sse2 => sse2::deblock_vertical(d, at, stride, n, size, t),
            _ => false,
        };
        // Scalar, or a vector kernel's 8-sample rows don't all fit in `d`.
        if !done {
            for i in 0..n {
                scalar::deblock_line(d, at + i * stride, 1, size, t);
            }
        }
    });
}

/// AV1 loop filter (7.14.6) of `lines` (a multiple of 4) lines across a horizontal edge with
/// the same filter: `d[base]` is q0 of the first line, the others follow it, and `stride` is
/// the distance between rows.
pub fn deblock_horizontal(d: &mut [u16], base: usize, stride: usize, lines: usize, size: LfSize, t: &LfThresh) {
    deblock_horizontal_at(level(), d, base, stride, lines, size, t);
}

/// [`deblock_horizontal`] on a given level.
pub fn deblock_horizontal_at(level: Level, d: &mut [u16], base: usize, stride: usize, lines: usize, size: LfSize, t: &LfThresh) {
    deblock_lines(lines, |first, n| {
        let at = base + first;
        let done = match level {
            #[cfg(target_arch = "aarch64")]
            Level::Neon => neon::deblock_horizontal(d, at, stride, n, size, t),
            #[cfg(target_arch = "x86_64")]
            Level::Sse2 => sse2::deblock_horizontal(d, at, stride, n, size, t),
            _ => false,
        };
        if !done {
            for i in 0..n {
                scalar::deblock_line(d, at + i, stride, size, t);
            }
        }
    });
}

/// `lines` in chunks for the vector kernels: (first line, 8 or 4 lines).
#[inline(always)]
fn deblock_lines(lines: usize, mut chunk: impl FnMut(usize, usize)) {
    let mut first = 0;
    while first + 4 <= lines {
        let n = if first + 8 <= lines { 8 } else { 4 };
        chunk(first, n);
        first += n;
    }
}

/// One row of the AV1 subpel filter's horizontal pass (7.11.3.4):
/// `out[j] = Round2(sum of taps[t] * src[j + t], shift)`, for as many `j` as `out` holds and
/// `src` has 7 more samples for. Samples are at most 12 bits, and the results must fit an i16
/// (true of AV1's filters after InterRound0).
pub fn subpel_h(src: &[u16], taps: &[i16; 8], out: &mut [i16], shift: u32) {
    subpel_h_at(level(), src, taps, out, shift);
}

/// [`subpel_h`] on a given level.
pub fn subpel_h_at(level: Level, src: &[u16], taps: &[i16; 8], out: &mut [i16], shift: u32) {
    // The kernels may then rely on `src` having `out.len() + 7` samples.
    let n = out.len().min(src.len().saturating_sub(7));
    let (Some(src), Some(out)) = (src.get(..(n + 7).min(src.len())), out.get_mut(..n)) else {
        return;
    };
    match level {
        #[cfg(target_arch = "aarch64")]
        Level::Neon => neon::subpel_h(src, taps, out, shift),
        #[cfg(target_arch = "x86_64")]
        Level::Sse2 => sse2::subpel_h(src, taps, out, shift),
        _ => scalar::subpel_h(src, taps, out, shift),
    }
}

/// One row of the AV1 subpel filter's vertical pass (7.11.3.4):
/// `out[j] = Round2(sum of taps[t] * rows[t * stride + j], shift)`; rows past the end of `rows`
/// count as zero.
pub fn subpel_v(rows: &[i16], stride: usize, taps: &[i16; 8], out: &mut [i32], shift: u32) {
    subpel_v_at(level(), rows, stride, taps, out, shift);
}

/// Whether `rows` holds 8 rows of `n` samples `stride` apart: the vector kernels need them all
/// (the scalar one reads missing rows as zero).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn whole_rows(rows: &[i16], stride: usize, n: usize) -> bool {
    n <= stride && stride.checked_mul(7).and_then(|s| s.checked_add(n)).is_some_and(|end| end <= rows.len())
}

/// [`subpel_v`] on a given level.
pub fn subpel_v_at(level: Level, rows: &[i16], stride: usize, taps: &[i16; 8], out: &mut [i32], shift: u32) {
    match level {
        #[cfg(target_arch = "aarch64")]
        Level::Neon if whole_rows(rows, stride, out.len()) => neon::subpel_v(rows, stride, taps, out, shift),
        #[cfg(target_arch = "x86_64")]
        Level::Sse2 if whole_rows(rows, stride, out.len()) => sse2::subpel_v(rows, stride, taps, out, shift),
        _ => scalar::subpel_v(rows, stride, taps, out, shift),
    }
}

/// Weight of each line sum's square in its direction's cost (7.15.2), lane by lane: 840 / n for
/// a line of n samples (`DIV_TABLE`), zero past the direction's last line.
const fn cdef_cost_weights() -> [[i32; 16]; 8] {
    const DIV: [i32; 9] = [0, 840, 420, 280, 210, 168, 140, 120, 105];
    let mut w = [[0i32; 16]; 8];
    let mut k = 0;
    while k < 8 {
        w[2][k] = DIV[8];
        w[6][k] = DIV[8];
        k += 1;
    }
    let mut i = 0;
    while i < 7 {
        w[0][i] = DIV[i + 1];
        w[0][14 - i] = DIV[i + 1];
        w[4][i] = DIV[i + 1];
        w[4][14 - i] = DIV[i + 1];
        i += 1;
    }
    w[0][7] = DIV[8];
    w[4][7] = DIV[8];
    let mut d = 1;
    while d < 8 {
        let mut j = 0;
        while j < 5 {
            w[d][3 + j] = DIV[8];
            j += 1;
        }
        let mut j = 0;
        while j < 3 {
            w[d][j] = DIV[2 * j + 2];
            w[d][10 - j] = DIV[2 * j + 2];
            j += 1;
        }
        d += 2;
    }
    w
}

static CDEF_COST_WEIGHTS: [[i32; 16]; 8] = cdef_cost_weights();

/// The best direction and its variance term from the eight costs (first maximum wins).
#[inline]
fn cdef_pick(cost: &[i32; 8]) -> (usize, i32) {
    let mut best = 0;
    let mut dir = 0;
    for (i, &c) in cost.iter().enumerate() {
        if c > best {
            best = c;
            dir = i;
        }
    }
    (dir, (best - cost[(dir + 4) & 7]) >> 10)
}

#[cfg(test)]
mod tests;
