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
