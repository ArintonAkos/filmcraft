//! NEON kernels (every aarch64 core). The arithmetic intrinsics are safe to call here because
//! the kernels are compiled with `neon` enabled; entering a kernel and loading or storing
//! through pointers need `unsafe`.

use core::arch::aarch64::*;

use crate::{CDEF_COST_WEIGHTS, cdef_pick};

/// 16-bit lanes; each line sum is two vectors (lanes 0..8 and 8..16), and a row is added at its
/// offset with `ext`, a fixed shift of lanes across two registers.
pub(crate) fn cdef_direction(rows: &[&[u16; 8]; 8], shift: u32) -> (usize, i32) {
    // SAFETY: NEON is part of the aarch64 baseline: every core this build runs on has it.
    unsafe { cdef_direction_neon(rows, shift) }
}

#[target_feature(enable = "neon")]
fn cdef_direction_neon(rows: &[&[u16; 8]; 8], shift: u32) -> (usize, i32) {
    let z = vdupq_n_s16(0);
    let down = vdupq_n_s16(-(shift as i16));
    let centre = vdupq_n_s16(128);
    let load = |r: &[u16; 8]| -> int16x8_t {
        // SAFETY: `r` is 8 u16, the 16 bytes `vld1q_u16` reads.
        let v = unsafe { vld1q_u16(r.as_ptr()) };
        vsubq_s16(vreinterpretq_s16_u16(vshlq_u16(v, down)), centre)
    };
    // Add `v` to the line sum (lo, hi) starting at lane `o`.
    macro_rules! at {
        ($lo:ident, $hi:ident, $v:expr, 0) => {{
            $lo = vaddq_s16($lo, $v);
        }};
        ($lo:ident, $hi:ident, $v:expr, $o:tt) => {{
            $lo = vaddq_s16($lo, vextq_s16::<{ 8 - $o }>(z, $v));
            $hi = vaddq_s16($hi, vextq_s16::<{ 8 - $o }>($v, z));
        }};
    }
    let (mut l0, mut h0, mut l1, mut h1, mut l3, mut h3, mut l4, mut h4) = (z, z, z, z, z, z, z, z);
    let (mut l5, mut h5, mut l7, mut h7, mut l6, mut l2) = (z, z, z, z, z, z);
    // Row i: directions 0, 1, 3 and 4 at offset i, 5 at 3 - i / 2, 7 at i / 2.
    macro_rules! row {
        ($i:tt, $o5:tt, $o7:tt) => {{
            let x = load(rows[$i]);
            let pairs = vpaddq_s16(x, z); // x0+x1, x2+x3, x4+x5, x6+x7, then zeros
            let rev_pairs = vrev64q_s16(pairs); // the four pairs reversed, then zeros
            let r = vrev64q_s16(x);
            let rev = vextq_s16::<4>(r, r); // x7 .. x0
            at!(l0, h0, x, $i);
            at!(l1, h1, pairs, $i);
            at!(l3, h3, rev_pairs, $i);
            at!(l4, h4, rev, $i);
            at!(l5, h5, x, $o5);
            at!(l7, h7, x, $o7);
            l6 = vaddq_s16(l6, x);
            l2 = vsetq_lane_s16::<$i>(vaddvq_s16(x), l2);
        }};
    }
    row!(0, 3, 0);
    row!(1, 3, 0);
    row!(2, 2, 1);
    row!(3, 2, 1);
    row!(4, 1, 2);
    row!(5, 1, 2);
    row!(6, 0, 3);
    row!(7, 0, 3);
    let lines = [(l0, h0), (l1, h1), (l2, z), (l3, h3), (l4, h4), (l5, h5), (l6, z), (l7, h7)];
    let cost: [i32; 8] = std::array::from_fn(|d| {
        let (lo, hi) = lines[d];
        let w = &CDEF_COST_WEIGHTS[d];
        // SAFETY: each pointer starts a run of 4 i32 inside the 16 of `w`.
        let (w0, w1, w2, w3) = unsafe { (vld1q_s32(w.as_ptr()), vld1q_s32(w.as_ptr().add(4)), vld1q_s32(w.as_ptr().add(8)), vld1q_s32(w.as_ptr().add(12))) };
        // Squares widened to 32 bits, times the lane weights, summed.
        let mut s = vmulq_s32(vmull_s16(vget_low_s16(lo), vget_low_s16(lo)), w0);
        s = vmlaq_s32(s, vmull_high_s16(lo, lo), w1);
        s = vmlaq_s32(s, vmull_s16(vget_low_s16(hi), vget_low_s16(hi)), w2);
        s = vmlaq_s32(s, vmull_high_s16(hi, hi), w3);
        vaddvq_s32(s)
    });
    cdef_pick(&cost)
}

// ---------------------------------------------------------------------------------------------
// Loop filter

/// 8 lanes of i16; the deblocking kernels use the first four.
type V = int16x8_t;

/// 4 lines across a horizontal edge (see [`crate::deblock_horizontal`]); `false` when the
/// samples aren't all in `d`.
pub(crate) fn deblock_horizontal(d: &mut [u16], base: usize, stride: usize, size: LfSize, t: &LfThresh) -> bool {
    // SAFETY: NEON is part of the aarch64 baseline: every core this build runs on has it.
    unsafe { horizontal(d, base, stride, size, t) }
}

/// 4 lines across a vertical edge (see [`crate::deblock_vertical`]); `false` when the samples
/// aren't all in `d`.
pub(crate) fn deblock_vertical(d: &mut [u16], base: usize, stride: usize, size: LfSize, t: &LfThresh) -> bool {
    // SAFETY: NEON is part of the aarch64 baseline: every core this build runs on has it.
    unsafe { vertical(d, base, stride, size, t) }
}

crate::deblock::deblock_kernels!("neon");

#[target_feature(enable = "neon")]
fn splat(v: i32) -> V {
    vdupq_n_s16(v as i16)
}
#[target_feature(enable = "neon")]
fn add(a: V, b: V) -> V {
    vaddq_s16(a, b)
}
#[target_feature(enable = "neon")]
fn sub(a: V, b: V) -> V {
    vsubq_s16(a, b)
}
#[target_feature(enable = "neon")]
fn abd(a: V, b: V) -> V {
    vabdq_s16(a, b)
}
#[target_feature(enable = "neon")]
fn gt(a: V, b: V) -> V {
    vreinterpretq_s16_u16(vcgtq_s16(a, b))
}
#[target_feature(enable = "neon")]
fn and(a: V, b: V) -> V {
    vandq_s16(a, b)
}
#[target_feature(enable = "neon")]
fn or(a: V, b: V) -> V {
    vorrq_s16(a, b)
}
#[target_feature(enable = "neon")]
fn andnot(a: V, b: V) -> V {
    vbicq_s16(b, a)
}
#[target_feature(enable = "neon")]
fn sel(m: V, a: V, b: V) -> V {
    vbslq_s16(vreinterpretq_u16_s16(m), a, b)
}
#[target_feature(enable = "neon")]
fn min(a: V, b: V) -> V {
    vminq_s16(a, b)
}
#[target_feature(enable = "neon")]
fn max(a: V, b: V) -> V {
    vmaxq_s16(a, b)
}
#[target_feature(enable = "neon")]
fn sra<const N: i32>(a: V) -> V {
    vshrq_n_s16::<N>(a)
}
#[target_feature(enable = "neon")]
fn srl<const N: i32>(a: V) -> V {
    vreinterpretq_s16_u16(vshrq_n_u16::<N>(vreinterpretq_u16_s16(a)))
}
#[target_feature(enable = "neon")]
fn any(m: V) -> bool {
    vmaxvq_u16(vreinterpretq_u16_s16(m)) != 0
}
#[target_feature(enable = "neon")]
fn lanes4() -> V {
    vcombine_s16(vdup_n_s16(-1), vdup_n_s16(0))
}
#[target_feature(enable = "neon")]
fn load4(s: &[u16; 4]) -> V {
    // SAFETY: `s` is 4 u16, the 8 bytes `vld1_u16` reads.
    let v = unsafe { vld1_u16(s.as_ptr()) };
    vreinterpretq_s16_u16(vcombine_u16(v, vdup_n_u16(0)))
}
#[target_feature(enable = "neon")]
fn store4(v: V, s: &mut [u16; 4]) {
    // SAFETY: `s` is 4 u16, the 8 bytes `vst1_u16` writes.
    unsafe { vst1_u16(s.as_mut_ptr(), vget_low_u16(vreinterpretq_u16_s16(v))) }
}
#[target_feature(enable = "neon")]
fn load8(s: &[u16; 8]) -> V {
    // SAFETY: `s` is 8 u16, the 16 bytes `vld1q_u16` reads.
    vreinterpretq_s16_u16(unsafe { vld1q_u16(s.as_ptr()) })
}
#[target_feature(enable = "neon")]
fn store8(v: V, s: &mut [u16; 8]) {
    // SAFETY: `s` is 8 u16, the 16 bytes `vst1q_u16` writes.
    unsafe { vst1q_u16(s.as_mut_ptr(), vreinterpretq_u16_s16(v)) }
}
/// 4 rows of 8 to 8 columns (in the first four lanes; the others hold column + 4 or column - 4).
#[target_feature(enable = "neon")]
fn transpose([r0, r1, r2, r3]: [V; 4]) -> [V; 8] {
    let (t0, t1) = (vtrn1q_s16(r0, r1), vtrn2q_s16(r0, r1));
    let (t2, t3) = (vtrn1q_s16(r2, r3), vtrn2q_s16(r2, r3));
    let w = vreinterpretq_s32_s16;
    let h = vreinterpretq_s16_s32;
    // c04: columns 0 and 4, c15, c26, c37.
    let c04 = h(vtrn1q_s32(w(t0), w(t2)));
    let c15 = h(vtrn1q_s32(w(t1), w(t3)));
    let c26 = h(vtrn2q_s32(w(t0), w(t2)));
    let c37 = h(vtrn2q_s32(w(t1), w(t3)));
    let up = |c: V| vextq_s16::<4>(c, c);
    [c04, c15, c26, c37, up(c04), up(c15), up(c26), up(c37)]
}
/// 8 columns (in the first four lanes) back to 4 rows of 8.
#[target_feature(enable = "neon")]
fn untranspose(c: [V; 8]) -> [V; 4] {
    let pair = |a: V, b: V| vreinterpretq_s32_s16(vcombine_s16(vget_low_s16(a), vget_low_s16(b)));
    let (c04, c15, c26, c37) = (pair(c[0], c[4]), pair(c[1], c[5]), pair(c[2], c[6]), pair(c[3], c[7]));
    let h = vreinterpretq_s16_s32;
    let (t0, t2) = (h(vtrn1q_s32(c04, c26)), h(vtrn2q_s32(c04, c26)));
    let (t1, t3) = (h(vtrn1q_s32(c15, c37)), h(vtrn2q_s32(c15, c37)));
    [vtrn1q_s16(t0, t1), vtrn2q_s16(t0, t1), vtrn1q_s16(t2, t3), vtrn2q_s16(t2, t3)]
}
