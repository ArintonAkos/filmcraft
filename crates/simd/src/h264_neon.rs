//! NEON kernels for H.264 8-bit inter prediction (8.4.2.2). Luma: 16 lanes of samples, the
//! 6-tap sums in 16-bit lanes (they fit: -2550..=10710), the centre position's second pass in
//! 32-bit lanes; the rounding narrowing shifts `vqrshrun` are exactly Clip1((v + 16) >> 5) and
//! Clip1((v + 512) >> 10). Chroma: the bilinear weights in 16-bit lanes.

use core::arch::aarch64::*;

/// Samples loaded per window row: 16, from up to 5 past the row start.
const REACH: usize = 5 + 16;

/// See [`crate::h264_luma`]; false when the window or `out` is too small for the vector loads
/// and stores (the caller then takes the scalar version).
pub(crate) fn luma(win: &[u8], ws: usize, fx: u32, fy: u32, w: usize, h: usize, out: &mut [u8], os: usize) -> bool {
    let fits = matches!(w, 4 | 8 | 16)
        && (1..=16).contains(&h)
        && (h + 4).checked_mul(ws).and_then(|n| n.checked_add(REACH)).is_some_and(|n| n <= win.len())
        && (h - 1).checked_mul(os).and_then(|n| n.checked_add(w)).is_some_and(|n| n <= out.len());
    if fits {
        // SAFETY: NEON is part of the aarch64 baseline: every core this build runs on has it.
        unsafe { luma_neon(win, ws, fx, fy, w, h, out, os) };
    }
    fits
}

/// 16 samples of `win` from `o`. Every offset the luma kernel loads from is at most
/// `(h + 4) * ws + 5` (window row h + 4, column 5), and `luma` checked `win` holds that plus 16.
#[target_feature(enable = "neon")]
fn ld(win: &[u8], o: usize) -> uint8x16_t {
    debug_assert!(o + 16 <= win.len());
    // SAFETY: `o + 16 <= win.len()` for every offset `luma_neon` passes (see above), so the 16
    // bytes `vld1q_u8` reads are inside `win`.
    unsafe { vld1q_u8(win.as_ptr().add(o)) }
}

/// The first `w` lanes of `v` into `out` from `o`.
#[target_feature(enable = "neon")]
fn put(out: &mut [u8], o: usize, w: usize, v: uint8x16_t) {
    let mut t = [0u8; 16];
    // SAFETY: `t` is 16 bytes, the 16 bytes `vst1q_u8` writes.
    unsafe { vst1q_u8(t.as_mut_ptr(), v) };
    match w {
        16 => copy::<16>(out, o, &t),
        8 => copy::<8>(out, o, &t),
        _ => copy::<4>(out, o, &t),
    }
}

#[inline(always)]
fn copy<const N: usize>(out: &mut [u8], o: usize, t: &[u8; 16]) {
    if let (Some(d), Some(s)) = (out.get_mut(o..).and_then(|d| d.first_chunk_mut::<N>()), t.first_chunk::<N>()) {
        *d = *s;
    }
}

/// a - 5b + 20c + 20d - 5e + f, unclipped, in two vectors of 8 lanes (wrapping u16 arithmetic
/// gives the exact signed result, which fits 16 bits).
#[target_feature(enable = "neon")]
fn tap6(v: [uint8x16_t; 6]) -> [int16x8_t; 2] {
    let [a, b, c, d, e, f] = v;
    let lo = vmlsq_n_u16(
        vmlaq_n_u16(vaddl_u8(vget_low_u8(a), vget_low_u8(f)), vaddl_u8(vget_low_u8(c), vget_low_u8(d)), 20),
        vaddl_u8(vget_low_u8(b), vget_low_u8(e)),
        5,
    );
    let hi = vmlsq_n_u16(vmlaq_n_u16(vaddl_high_u8(a, f), vaddl_high_u8(c, d), 20), vaddl_high_u8(b, e), 5);
    [vreinterpretq_s16_u16(lo), vreinterpretq_s16_u16(hi)]
}

/// The horizontal 6-tap of window row `r` (outputs from its column 0's window).
#[target_feature(enable = "neon")]
fn tap6_h(win: &[u8], ws: usize, r: usize) -> [int16x8_t; 2] {
    let o = r * ws;
    tap6([ld(win, o), ld(win, o + 1), ld(win, o + 2), ld(win, o + 3), ld(win, o + 4), ld(win, o + 5)])
}

/// The vertical 6-tap of window rows r..r + 6 at column `c`.
#[target_feature(enable = "neon")]
fn tap6_v(win: &[u8], ws: usize, r: usize, c: usize) -> [int16x8_t; 2] {
    tap6([0, 1, 2, 3, 4, 5].map(|k| ld(win, (r + k) * ws + c)))
}

/// Clip1((t + 16) >> 5).
#[target_feature(enable = "neon")]
fn half(t: [int16x8_t; 2]) -> uint8x16_t {
    vcombine_u8(vqrshrun_n_s16::<5>(t[0]), vqrshrun_n_s16::<5>(t[1]))
}

/// The centre sample j: Clip1((6-tap of the unclipped horizontal half samples of rows
/// k[0..6] + 512) >> 10), in 32-bit lanes.
#[target_feature(enable = "neon")]
fn centre(k: &[[int16x8_t; 2]]) -> uint8x16_t {
    let mut r = [vdup_n_u8(0); 2];
    for (p, r) in r.iter_mut().enumerate() {
        let t = |i: usize| k.get(i).map_or(vdupq_n_s16(0), |v| v[p]);
        let (t0, t1, t2, t3, t4, t5) = (t(0), t(1), t(2), t(3), t(4), t(5));
        let lo = vmlsl_n_s16(
            vmlsl_n_s16(
                vmlal_n_s16(vmlal_n_s16(vaddl_s16(vget_low_s16(t0), vget_low_s16(t5)), vget_low_s16(t2), 20), vget_low_s16(t3), 20),
                vget_low_s16(t1),
                5,
            ),
            vget_low_s16(t4),
            5,
        );
        let hi = vmlsl_high_n_s16(vmlsl_high_n_s16(vmlal_high_n_s16(vmlal_high_n_s16(vaddl_high_s16(t0, t5), t2, 20), t3, 20), t1, 5), t4, 5);
        *r = vqmovn_u16(vcombine_u16(vqrshrun_n_s32::<10>(lo), vqrshrun_n_s32::<10>(hi)));
    }
    vcombine_u8(r[0], r[1])
}

#[target_feature(enable = "neon")]
fn luma_neon(win: &[u8], ws: usize, fx: u32, fy: u32, w: usize, h: usize, out: &mut [u8], os: usize) {
    // Sample G of block row r is window row r + 2, column 2.
    let g = |r: usize, dr: usize, dc: usize| ld(win, (r + 2 + dr) * ws + 2 + dc);
    match (fx, fy) {
        (0, 0) => {
            for r in 0..h {
                put(out, r * os, w, g(r, 0, 0));
            }
        }
        (_, 0) => {
            for r in 0..h {
                let b = half(tap6_h(win, ws, r + 2));
                let v = match fx {
                    2 => b,
                    1 => vrhaddq_u8(g(r, 0, 0), b),
                    _ => vrhaddq_u8(g(r, 0, 1), b),
                };
                put(out, r * os, w, v);
            }
        }
        (0, _) => {
            for r in 0..h {
                let hv = half(tap6_v(win, ws, r, 2));
                let v = match fy {
                    2 => hv,
                    1 => vrhaddq_u8(g(r, 0, 0), hv),
                    _ => vrhaddq_u8(g(r, 1, 0), hv),
                };
                put(out, r * os, w, v);
            }
        }
        (2, _) | (_, 2) => {
            // Unclipped horizontal half samples of window rows 0..h + 5, once per block.
            let mut b1 = [[vdupq_n_s16(0); 2]; 21];
            for (k, b) in b1.iter_mut().enumerate().take(h + 5) {
                *b = tap6_h(win, ws, k);
            }
            for r in 0..h {
                let k = b1.get(r..r + 6).unwrap_or(&[]);
                let j = centre(k);
                let v = match (fx, fy) {
                    (2, 2) => j,
                    (2, 1) => vrhaddq_u8(half(b1[r + 2]), j),
                    (2, _) => vrhaddq_u8(half(b1[r + 3]), j),
                    _ => vrhaddq_u8(half(tap6_v(win, ws, r, if fx == 1 { 2 } else { 3 })), j),
                };
                put(out, r * os, w, v);
            }
        }
        _ => {
            // e, g, p, r: a horizontal half sample of row r + 2 or r + 3 and a vertical one of
            // column c + 2 or c + 3.
            let hoff = if fy == 1 { 2 } else { 3 };
            let voff = if fx == 1 { 2 } else { 3 };
            for r in 0..h {
                put(out, r * os, w, vrhaddq_u8(half(tap6_h(win, ws, r + hoff)), half(tap6_v(win, ws, r, voff))));
            }
        }
    }
}

/// See [`crate::h264_chroma`]; false when the window or `out` is too small.
pub(crate) fn chroma(win: &[u8], ws: usize, fx: u32, fy: u32, w: usize, h: usize, out: &mut [u8], os: usize) -> bool {
    let fits = matches!(w, 2 | 4 | 8)
        && (1..=8).contains(&h)
        && fx < 8
        && fy < 8
        && h.checked_mul(ws).and_then(|n| n.checked_add(9)).is_some_and(|n| n <= win.len())
        && (h - 1).checked_mul(os).and_then(|n| n.checked_add(w)).is_some_and(|n| n <= out.len());
    if fits {
        // SAFETY: NEON is part of the aarch64 baseline: every core this build runs on has it.
        unsafe { chroma_neon(win, ws, fx, fy, w, h, out, os) };
    }
    fits
}

#[inline(always)]
fn copy8<const N: usize>(out: &mut [u8], o: usize, t: &[u8; 8]) {
    if let (Some(d), Some(s)) = (out.get_mut(o..).and_then(|d| d.first_chunk_mut::<N>()), t.first_chunk::<N>()) {
        *d = *s;
    }
}

/// 8 samples of `win` from `o` (zeros past its end, which `chroma` rules out).
#[target_feature(enable = "neon")]
fn ld8(win: &[u8], o: usize) -> uint8x8_t {
    match win.get(o..).and_then(|s| s.first_chunk::<8>()) {
        // SAFETY: `s` is 8 bytes, the 8 bytes `vld1_u8` reads.
        Some(s) => unsafe { vld1_u8(s.as_ptr()) },
        None => vdup_n_u8(0),
    }
}

#[target_feature(enable = "neon")]
fn chroma_neon(win: &[u8], ws: usize, fx: u32, fy: u32, w: usize, h: usize, out: &mut [u8], os: usize) {
    // Weights at most 64: the sums stay below 2^14.
    let [a, b, c, d] = [(8 - fx) * (8 - fy), fx * (8 - fy), (8 - fx) * fy, fx * fy].map(|v| vdup_n_u8(v as u8));
    for r in 0..h {
        let (o0, o1) = (r * ws, (r + 1) * ws);
        let acc = vmlal_u8(vmlal_u8(vmlal_u8(vmull_u8(ld8(win, o0), a), ld8(win, o0 + 1), b), ld8(win, o1), c), ld8(win, o1 + 1), d);
        let v = vrshrn_n_u16::<6>(acc);
        let mut t = [0u8; 8];
        // SAFETY: `t` is 8 bytes, the 8 bytes `vst1_u8` writes.
        unsafe { vst1_u8(t.as_mut_ptr(), v) };
        let o = r * os;
        match w {
            8 => copy8::<8>(out, o, &t),
            4 => copy8::<4>(out, o, &t),
            _ => copy8::<2>(out, o, &t),
        }
    }
}

/// See [`crate::h264_deblock`]; false when the edge's samples do not fit `pix`.
pub(crate) fn deblock(pix: &mut [u8], stride: usize, x: usize, y: usize, vertical: bool, chroma: bool, e: &crate::H264Edge) -> bool {
    let n = if chroma { 8 } else { 16 };
    // The last byte read or written: a vertical edge reads lines y .. y + n at x - 4 .. x + 4, a
    // horizontal one rows y - 4 .. y + 4 at x .. x + n.
    let end = if vertical {
        if x < 4 { None } else { (y + n - 1).checked_mul(stride).and_then(|o| o.checked_add(x + 4)) }
    } else if y < 4 {
        None
    } else {
        (y + 3).checked_mul(stride).and_then(|o| o.checked_add(x + n))
    };
    let fits = end.is_some_and(|end| end <= pix.len());
    if fits {
        // SAFETY: NEON is part of the aarch64 baseline: every core this build runs on has it.
        unsafe { deblock_neon(pix, stride, x, y, vertical, chroma, e) };
    }
    fits
}

/// 8 bytes of `pix` from `o`; `deblock` checked every offset `deblock_neon` passes.
#[target_feature(enable = "neon")]
fn px8(pix: &[u8], o: usize) -> uint8x8_t {
    debug_assert!(o + 8 <= pix.len());
    // SAFETY: `o + 8 <= pix.len()` (checked by `deblock` for every row of the edge).
    unsafe { vld1_u8(pix.as_ptr().add(o)) }
}

#[target_feature(enable = "neon")]
fn px16(pix: &[u8], o: usize) -> uint8x16_t {
    debug_assert!(o + 16 <= pix.len());
    // SAFETY: `o + 16 <= pix.len()` (checked by `deblock` for every row of a luma edge).
    unsafe { vld1q_u8(pix.as_ptr().add(o)) }
}

#[target_feature(enable = "neon")]
fn put8(pix: &mut [u8], o: usize, v: uint8x8_t) {
    debug_assert!(o + 8 <= pix.len());
    // SAFETY: as `px8`.
    unsafe { vst1_u8(pix.as_mut_ptr().add(o), v) }
}

#[target_feature(enable = "neon")]
fn put16(pix: &mut [u8], o: usize, v: uint8x16_t) {
    debug_assert!(o + 16 <= pix.len());
    // SAFETY: as `px16`.
    unsafe { vst1q_u8(pix.as_mut_ptr().add(o), v) }
}

/// The 8x8 byte transpose (its own inverse): row i of the result is column i of `a`.
#[target_feature(enable = "neon")]
fn transpose8(a: [uint8x8_t; 8]) -> [uint8x8_t; 8] {
    let b = [
        vtrn1_u8(a[0], a[1]),
        vtrn2_u8(a[0], a[1]),
        vtrn1_u8(a[2], a[3]),
        vtrn2_u8(a[2], a[3]),
        vtrn1_u8(a[4], a[5]),
        vtrn2_u8(a[4], a[5]),
        vtrn1_u8(a[6], a[7]),
        vtrn2_u8(a[6], a[7]),
    ];
    let h = b.map(|v| vreinterpret_u16_u8(v));
    let c = [
        vtrn1_u16(h[0], h[2]),
        vtrn1_u16(h[1], h[3]),
        vtrn2_u16(h[0], h[2]),
        vtrn2_u16(h[1], h[3]),
        vtrn1_u16(h[4], h[6]),
        vtrn1_u16(h[5], h[7]),
        vtrn2_u16(h[4], h[6]),
        vtrn2_u16(h[5], h[7]),
    ];
    let w = c.map(|v| vreinterpret_u32_u16(v));
    let d = [
        vtrn1_u32(w[0], w[4]),
        vtrn1_u32(w[1], w[5]),
        vtrn1_u32(w[2], w[6]),
        vtrn1_u32(w[3], w[7]),
        vtrn2_u32(w[0], w[4]),
        vtrn2_u32(w[1], w[5]),
        vtrn2_u32(w[2], w[6]),
        vtrn2_u32(w[3], w[7]),
    ];
    d.map(|v| vreinterpret_u8_u32(v))
}

/// Round(weighted sum of `t` / 2^SH) of samples, the sums in 16-bit lanes (at most 8 * 255).
#[target_feature(enable = "neon")]
fn wsum<const SH: i32>(t: &[(uint8x16_t, u8)]) -> uint8x16_t {
    let (mut lo, mut hi) = (vdupq_n_u16(0), vdupq_n_u16(0));
    for &(v, w) in t {
        lo = vmlal_u8(lo, vget_low_u8(v), vdup_n_u8(w));
        hi = vmlal_high_u8(hi, v, vdupq_n_u8(w));
    }
    vcombine_u8(vrshrn_n_u16::<SH>(lo), vrshrn_n_u16::<SH>(hi))
}

/// Clip3(-tc, tc, ((q0 - p0) * 4 + (p1 - q1) + 4) >> 3), in 16-bit lanes, narrowed (|tc| < 28).
#[target_feature(enable = "neon")]
fn delta(p1: uint8x16_t, p0: uint8x16_t, q0: uint8x16_t, q1: uint8x16_t, tc: uint8x16_t) -> int8x16_t {
    let half = |qp: uint16x8_t, pq: uint16x8_t, tc: uint16x8_t| {
        let d = vshrq_n_s16::<3>(vaddq_s16(vaddq_s16(vshlq_n_s16::<2>(vreinterpretq_s16_u16(qp)), vreinterpretq_s16_u16(pq)), vdupq_n_s16(4)));
        let tc = vreinterpretq_s16_u16(tc);
        vmovn_s16(vminq_s16(vmaxq_s16(d, vnegq_s16(tc)), tc))
    };
    let lo = half(vsubl_u8(vget_low_u8(q0), vget_low_u8(p0)), vsubl_u8(vget_low_u8(p1), vget_low_u8(q1)), vmovl_u8(vget_low_u8(tc)));
    let hi = half(vsubl_high_u8(q0, p0), vsubl_high_u8(p1, q1), vmovl_high_u8(tc));
    vcombine_s8(lo, hi)
}

/// [`crate::h264::filter_lanes`] on 16 lanes of bytes (chroma: the first 8).
#[target_feature(enable = "neon")]
fn filter(s: &mut [uint8x16_t; 8], e: &crate::H264Edge, chroma: bool) {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = *s;
    let per = if chroma { 2 } else { 4 };
    let (mut on_b, mut tc_b) = ([0u8; 16], [0u8; 16]);
    for i in 0..16 {
        if let (Some(&b), Some(&t)) = (e.bs.get(i / per), e.tc0.get(i / per)) {
            on_b[i] = if b != 0 { 0xff } else { 0 };
            tc_b[i] = t;
        }
    }
    // SAFETY: `on_b` and `tc_b` are 16 bytes, the 16 bytes `vld1q_u8` reads.
    let (bs_on, tc0) = unsafe { (vld1q_u8(on_b.as_ptr()), vld1q_u8(tc_b.as_ptr())) };
    let (alpha, beta) = (vdupq_n_u8(e.alpha), vdupq_n_u8(e.beta));
    let on = vandq_u8(vandq_u8(bs_on, vcltq_u8(vabdq_u8(p0, q0), alpha)), vandq_u8(vcltq_u8(vabdq_u8(p1, p0), beta), vcltq_u8(vabdq_u8(q1, q0), beta)));
    if vmaxvq_u8(on) == 0 {
        return;
    }
    // bS 4 is only ever assigned to all four segments of an edge.
    let strong_edge = e.bs[0] >= 4;
    if chroma {
        let (np0, nq0) = if strong_edge {
            (wsum::<2>(&[(p1, 2), (p0, 1), (q1, 1)]), wsum::<2>(&[(q1, 2), (q0, 1), (p1, 1)]))
        } else {
            let d = delta(p1, p0, q0, q1, vaddq_u8(tc0, vdupq_n_u8(1)));
            (vsqaddq_u8(p0, d), vsqaddq_u8(q0, vnegq_s8(d)))
        };
        s[3] = vbslq_u8(on, np0, p0);
        s[4] = vbslq_u8(on, nq0, q0);
        return;
    }
    let ap = vcltq_u8(vabdq_u8(p2, p0), beta);
    let aq = vcltq_u8(vabdq_u8(q2, q0), beta);
    if strong_edge {
        let strong = vcltq_u8(vabdq_u8(p0, q0), vdupq_n_u8((e.alpha >> 2) + 2));
        let sp = vandq_u8(vandq_u8(on, ap), strong);
        let sq = vandq_u8(vandq_u8(on, aq), strong);
        let wp0 = wsum::<2>(&[(p1, 2), (p0, 1), (q1, 1)]);
        let wq0 = wsum::<2>(&[(q1, 2), (q0, 1), (p1, 1)]);
        s[3] = vbslq_u8(sp, wsum::<3>(&[(p2, 1), (p1, 2), (p0, 2), (q0, 2), (q1, 1)]), vbslq_u8(on, wp0, p0));
        s[2] = vbslq_u8(sp, wsum::<2>(&[(p2, 1), (p1, 1), (p0, 1), (q0, 1)]), p1);
        s[1] = vbslq_u8(sp, wsum::<3>(&[(p3, 2), (p2, 3), (p1, 1), (p0, 1), (q0, 1)]), p2);
        s[4] = vbslq_u8(sq, wsum::<3>(&[(p1, 1), (p0, 2), (q0, 2), (q1, 2), (q2, 1)]), vbslq_u8(on, wq0, q0));
        s[5] = vbslq_u8(sq, wsum::<2>(&[(p0, 1), (q0, 1), (q1, 1), (q2, 1)]), q1);
        s[6] = vbslq_u8(sq, wsum::<3>(&[(q3, 2), (q2, 3), (q1, 1), (q0, 1), (p0, 1)]), q2);
    } else {
        // tc = tC0 + (ap ? 1 : 0) + (aq ? 1 : 0): the masks are 0xff, so subtracting adds 1.
        let tc = vsubq_u8(vsubq_u8(tc0, ap), aq);
        let d = delta(p1, p0, q0, q1, tc);
        let avg = vrhaddq_u8(p0, q0);
        // p1 + Clip3(-tC0, tC0, (p2 + avg - 2 * p1) >> 1) = Clip3(p1 - tC0, p1 + tC0, (p2 + avg) >> 1).
        let np1 = vminq_u8(vmaxq_u8(vhaddq_u8(p2, avg), vqsubq_u8(p1, tc0)), vqaddq_u8(p1, tc0));
        let nq1 = vminq_u8(vmaxq_u8(vhaddq_u8(q2, avg), vqsubq_u8(q1, tc0)), vqaddq_u8(q1, tc0));
        s[3] = vbslq_u8(on, vsqaddq_u8(p0, d), p0);
        s[4] = vbslq_u8(on, vsqaddq_u8(q0, vnegq_s8(d)), q0);
        s[2] = vbslq_u8(vandq_u8(on, ap), np1, p1);
        s[5] = vbslq_u8(vandq_u8(on, aq), nq1, q1);
    }
}

#[target_feature(enable = "neon")]
fn deblock_neon(pix: &mut [u8], stride: usize, x: usize, y: usize, vertical: bool, chroma: bool, e: &crate::H264Edge) {
    let zero = vdup_n_u8(0);
    let mut s = [vdupq_n_u8(0); 8];
    if vertical {
        // Line i's samples p3 .. q3 are the 8 bytes from x - 4 of row y + i: transposed, vector k
        // holds sample k of every line.
        let rows = |i0: usize| transpose8(std::array::from_fn(|i| px8(pix, (y + i0 + i) * stride + x - 4)));
        let lo = rows(0);
        let hi = if chroma { [zero; 8] } else { rows(8) };
        for k in 0..8 {
            s[k] = vcombine_u8(lo[k], hi[k]);
        }
        filter(&mut s, e, chroma);
        let lo = transpose8(s.map(|v| vget_low_u8(v)));
        for (i, v) in lo.into_iter().enumerate() {
            put8(pix, (y + i) * stride + x - 4, v);
        }
        if !chroma {
            let hi = transpose8(s.map(|v| vget_high_u8(v)));
            for (i, v) in hi.into_iter().enumerate() {
                put8(pix, (y + 8 + i) * stride + x - 4, v);
            }
        }
    } else {
        let base = (y - 4) * stride + x;
        for (k, v) in s.iter_mut().enumerate() {
            *v = if chroma { vcombine_u8(px8(pix, base + k * stride), zero) } else { px16(pix, base + k * stride) };
        }
        filter(&mut s, e, chroma);
        // Only p2 .. q2 (luma) or p0, q0 (chroma) can change.
        if chroma {
            for k in 3..5 {
                put8(pix, base + k * stride, vget_low_u8(s[k]));
            }
        } else {
            for k in 1..7 {
                put16(pix, base + k * stride, s[k]);
            }
        }
    }
}
