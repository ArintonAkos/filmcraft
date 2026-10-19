//! SSE2 kernels (every x86_64 core). The arithmetic intrinsics are safe to call here because
//! the kernels are compiled with `sse2` enabled; entering a kernel and loading or storing
//! through pointers need `unsafe`.

use core::arch::x86_64::*;

use crate::{CDEF_COST_WEIGHTS, cdef_pick};

/// 16-bit lanes; each line sum is two vectors (lanes 0..8 and 8..16), and a row is added at its
/// offset with byte shifts that move it up across the two registers.
pub(crate) fn cdef_direction(rows: &[&[u16; 8]; 8], shift: u32) -> (usize, i32) {
    // SAFETY: SSE2 is part of the x86_64 baseline: every core this build runs on has it.
    unsafe { cdef_direction_sse2(rows, shift) }
}

#[target_feature(enable = "sse2")]
fn cdef_direction_sse2(rows: &[&[u16; 8]; 8], shift: u32) -> (usize, i32) {
    let z = _mm_setzero_si128();
    let down = _mm_cvtsi32_si128(shift as i32);
    let centre = _mm_set1_epi16(128);
    let ones = _mm_set1_epi16(1);
    let load = |r: &[u16; 8]| -> __m128i {
        // SAFETY: `r` is 8 u16, the 16 bytes `_mm_loadu_si128` reads (unaligned load).
        let v = unsafe { _mm_loadu_si128(r.as_ptr().cast()) };
        _mm_sub_epi16(_mm_srl_epi16(v, down), centre)
    };
    // Add `v` to the line sum (lo, hi) starting at lane `o`.
    macro_rules! at {
        ($lo:ident, $hi:ident, $v:expr, 0) => {{
            $lo = _mm_add_epi16($lo, $v);
        }};
        ($lo:ident, $hi:ident, $v:expr, $o:tt) => {{
            $lo = _mm_add_epi16($lo, _mm_slli_si128::<{ 2 * $o }>($v));
            $hi = _mm_add_epi16($hi, _mm_srli_si128::<{ 16 - 2 * $o }>($v));
        }};
    }
    // The four lanes of 32-bit sums added together.
    let total = |m: __m128i| -> i32 {
        let s = _mm_add_epi32(m, _mm_shuffle_epi32::<0x4E>(m));
        _mm_cvtsi128_si32(_mm_add_epi32(s, _mm_shuffle_epi32::<0xB1>(s)))
    };
    let (mut l0, mut h0, mut l1, mut h1, mut l3, mut h3, mut l4, mut h4) = (z, z, z, z, z, z, z, z);
    let (mut l5, mut h5, mut l7, mut h7, mut l6, mut l2) = (z, z, z, z, z, z);
    // Row i: directions 0, 1, 3 and 4 at offset i, 5 at 3 - i / 2, 7 at i / 2.
    macro_rules! row {
        ($i:tt, $o5:tt, $o7:tt) => {{
            let x = load(rows[$i]);
            let sums = _mm_madd_epi16(x, ones); // x0+x1, x2+x3, x4+x5, x6+x7 as i32
            let pairs = _mm_packs_epi32(sums, z); // the pairs as i16, then zeros (|pair| <= 256)
            let rev_pairs = _mm_shufflelo_epi16::<0x1B>(pairs);
            let rev = _mm_shuffle_epi32::<0x4E>(_mm_shufflehi_epi16::<0x1B>(_mm_shufflelo_epi16::<0x1B>(x)));
            at!(l0, h0, x, $i);
            at!(l1, h1, pairs, $i);
            at!(l3, h3, rev_pairs, $i);
            at!(l4, h4, rev, $i);
            at!(l5, h5, x, $o5);
            at!(l7, h7, x, $o7);
            l6 = _mm_add_epi16(l6, x);
            l2 = _mm_insert_epi16::<$i>(l2, total(sums));
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
        let mut p = [0i16; 16];
        // SAFETY: `p` is 16 i16; each store writes 16 bytes inside it (unaligned stores).
        unsafe {
            _mm_storeu_si128(p.as_mut_ptr().cast(), lo);
            _mm_storeu_si128(p.as_mut_ptr().add(8).cast(), hi);
        }
        p.iter().zip(&CDEF_COST_WEIGHTS[d]).map(|(&v, &w)| v as i32 * v as i32 * w).sum()
    });
    cdef_pick(&cost)
}
