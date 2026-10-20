//! AV1 loop filter (deblocking) sample filters (7.14.6): the types the API takes, and the vector
//! kernels written once as a macro that each architecture module expands over its own vector
//! operations.

/// The sample filter an edge uses: 7.14.3 `filterSize` 4, 8 or 16, and on chroma planes 8 is
/// the 6-tap filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LfSize {
    Four,
    Six,
    Eight,
    Sixteen,
}

impl LfSize {
    /// The filter for an edge of `filter_size` (4, 8 or 16) on a luma or chroma plane.
    pub fn new(filter_size: usize, chroma: bool) -> LfSize {
        match (filter_size, chroma) {
            (0..=4, _) => LfSize::Four,
            (_, true) => LfSize::Six,
            (5..=8, false) => LfSize::Eight,
            _ => LfSize::Sixteen,
        }
    }

    /// Samples the filter reads on each side of the edge.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    pub(crate) const fn reach(self) -> usize {
        match self {
            LfSize::Four => 2,
            LfSize::Six => 3,
            LfSize::Eight => 4,
            LfSize::Sixteen => 7,
        }
    }

    /// Samples the filter can change on each side of the edge.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    pub(crate) const fn changes(self) -> usize {
        match self {
            LfSize::Four | LfSize::Six => 2,
            LfSize::Eight => 3,
            LfSize::Sixteen => 6,
        }
    }
}

/// Per-edge thresholds of the sample filter, scaled to the bit depth.
#[derive(Clone, Copy, Debug)]
pub struct LfThresh {
    pub(crate) limit: i32,
    pub(crate) blimit: i32,
    pub(crate) thresh: i32,
    /// Flatness threshold 1 << (BitDepth - 8).
    pub(crate) flat: i32,
    /// 0x80 << (BitDepth - 8) and 1 << (BitDepth - 1).
    pub(crate) off: i32,
    pub(crate) half: i32,
}

impl LfThresh {
    /// The thresholds of an edge with filter strength `limit`, `blimit` and `thresh` (7.14.4)
    /// in a `bit_depth` (8 to 12) picture.
    pub fn new(limit: i32, blimit: i32, thresh: i32, bit_depth: u32) -> LfThresh {
        let sh = bit_depth.clamp(8, 12) - 8;
        LfThresh { limit: limit << sh, blimit: blimit << sh, thresh: thresh << sh, flat: 1 << sh, off: 0x80 << sh, half: 1 << (sh + 7) }
    }
}

/// The vector kernels over 8 lanes of i16, one line across the edge per lane (8 lines, or 4 in
/// the first lanes). Samples (at most 12 bits) fit an i16; the wide filters' weighted sums (at
/// most 16 * 4095 + 8) fit 16 bits unsigned, so they add with wrapping and shift right
/// logically.
///
/// The expanding module provides the type `V`, the target feature `$feature`, and these
/// functions, all compiled with that feature: `splat`, `add`, `sub` (wrapping), `abd` (absolute
/// difference of non-negative lanes), `gt` (all ones where a > b), `and`, `or`, `andnot`
/// (!a & b), `sel` (m ? a : b), `min`, `max`, `sra::<N>`, `srl::<N>`, `any` (a lane set),
/// `lanes4` (the first four lanes set), `load4`/`store4` (4 samples into the first lanes),
/// `load8`/`store8`, and `transpose` (8 rows of 8 samples to 8 columns: its own inverse).
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
macro_rules! deblock_kernels {
    ($feature:literal) => {
        use $crate::deblock::{LfSize, LfThresh};

        /// Where tap `k` (offset `k - 8` from q0) of a line starting at `base` lies.
        fn tap(base: usize, k: usize, step: usize) -> Option<usize> {
            base.checked_add(k.checked_mul(step)?)?.checked_sub(step.checked_mul(8)?)
        }

        /// `n` (4 or 8) lines across a horizontal edge: q0 of the first at `d[base]`, the
        /// others after it, `stride` between rows. `false` (nothing changed) when the samples
        /// it reads aren't all in `d`.
        #[target_feature(enable = $feature)]
        fn horizontal(d: &mut [u16], base: usize, stride: usize, n: usize, size: LfSize, t: &LfThresh) -> bool {
            let (r, w) = (size.reach(), size.changes());
            let mut x = [splat(0); 16];
            for k in 8 - r..8 + r {
                let Some(row) = tap(base, k, stride).and_then(|i| d.get(i..i.checked_add(n)?)) else {
                    return false;
                };
                x[k] = match (row.first_chunk::<8>(), row.first_chunk::<4>()) {
                    (Some(row), _) => load8(row),
                    (None, Some(row)) => load4(row),
                    _ => return false,
                };
            }
            filter(&mut x, size, t, if n == 8 { splat(-1) } else { lanes4() });
            for k in 8 - w..8 + w {
                let mut row = tap(base, k, stride).and_then(|i| d.get_mut(i..i.checked_add(n)?));
                if let Some(row) = row.as_deref_mut().and_then(|s| s.first_chunk_mut::<8>()) {
                    store8(x[k], row);
                } else if let Some(row) = row.and_then(|s| s.first_chunk_mut::<4>()) {
                    store4(x[k], row);
                }
            }
            true
        }

        /// `n` (4 or 8) lines across a vertical edge: q0 of the first at `d[base]`, the others
        /// one `stride` below each other. Reads and writes 8 samples per line (16 for the
        /// 16-wide filter) centred on the edge; `false` (nothing changed) when they aren't all
        /// in `d`.
        #[target_feature(enable = $feature)]
        fn vertical(d: &mut [u16], base: usize, stride: usize, n: usize, size: LfSize, t: &LfThresh) -> bool {
            // The first tap of each 8-sample run the filter needs.
            let runs: &[usize] = if size == LfSize::Sixteen { &[0, 8] } else { &[4] };
            let row_at = |i: usize, k: usize| tap(base, k, 1)?.checked_add(i.checked_mul(stride)?);
            let mut x = [splat(0); 16];
            for &k in runs {
                let mut rows = [splat(0); 8];
                for (i, v) in rows[..n].iter_mut().enumerate() {
                    let Some(s) = row_at(i, k).and_then(|j| d.get(j..)).and_then(|s| s.first_chunk::<8>()) else {
                        return false;
                    };
                    *v = load8(s);
                }
                x[k..k + 8].copy_from_slice(&transpose(rows));
            }
            filter(&mut x, size, t, if n == 8 { splat(-1) } else { lanes4() });
            for &k in runs {
                let mut cols = [splat(0); 8];
                cols.copy_from_slice(&x[k..k + 8]);
                for (i, v) in transpose(cols).into_iter().take(n).enumerate() {
                    if let Some(s) = row_at(i, k).and_then(|j| d.get_mut(j..)).and_then(|s| s.first_chunk_mut::<8>()) {
                        store8(v, s);
                    }
                }
            }
            true
        }

        #[target_feature(enable = $feature)]
        fn clamp(v: V, lo: V, hi: V) -> V {
            max(min(v, hi), lo)
        }

        /// The sample filter (7.14.6) on taps `x[k]` at offset `k - 8` from q0, one line per lane
        /// (the lanes set in `lines`). Every lane runs every filter and keeps the result its
        /// masks pick.
        #[target_feature(enable = $feature)]
        fn filter(x: &mut [V; 16], size: LfSize, t: &LfThresh, lines: V) {
            let (lim, tf) = (splat(t.limit), splat(t.flat));
            let [p3, p2, p1, p0, q0, q1, q2, q3] = [x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11]];
            // filter mask (7.14.6.2)
            let (ap, aq, a0) = (abd(p1, p0), abd(q1, q0), abd(p0, q0));
            let mut fail = or(or(gt(ap, lim), gt(aq, lim)), gt(add(add(a0, a0), srl::<1>(abd(p1, q1))), splat(t.blimit)));
            let mut flat = splat(0);
            if size != LfSize::Four {
                fail = or(fail, or(gt(abd(p2, p1), lim), gt(abd(q2, q1), lim)));
                let mut rough = or(or(gt(ap, tf), gt(aq, tf)), or(gt(abd(p2, p0), tf), gt(abd(q2, q0), tf)));
                if size != LfSize::Six {
                    fail = or(fail, or(gt(abd(p3, p2), lim), gt(abd(q3, q2), lim)));
                    rough = or(rough, or(gt(abd(p3, p0), tf), gt(abd(q3, q0), tf)));
                }
                flat = andnot(rough, splat(-1));
            }
            let on = andnot(fail, lines);
            if !any(on) {
                return;
            }
            let flat = and(flat, on);
            let narrow = andnot(flat, on);
            if any(narrow) {
                // narrow filter (7.14.6.3)
                let hev = or(gt(ap, splat(t.thresh)), gt(aq, splat(t.thresh)));
                let (lo, hi, off) = (splat(-t.half), splat(t.half - 1), splat(t.off));
                let (ps1, ps0, qs0, qs1) = (sub(p1, off), sub(p0, off), sub(q0, off), sub(q1, off));
                let mut f = and(hev, clamp(sub(ps1, qs1), lo, hi));
                let d0 = sub(qs0, ps0);
                f = clamp(add(f, add(d0, add(d0, d0))), lo, hi);
                let f1 = sra::<3>(clamp(add(f, splat(4)), lo, hi));
                let f2 = sra::<3>(clamp(add(f, splat(3)), lo, hi));
                x[8] = sel(narrow, add(clamp(sub(qs0, f1), lo, hi), off), q0);
                x[7] = sel(narrow, add(clamp(add(ps0, f2), lo, hi), off), p0);
                let outer = andnot(hev, narrow);
                let f = sra::<1>(add(f1, splat(1)));
                x[9] = sel(outer, add(clamp(sub(qs1, f), lo, hi), off), q1);
                x[6] = sel(outer, add(clamp(add(ps1, f), lo, hi), off), p1);
            }
            // The wide filters read the flat lanes, which the narrow filter left alone.
            if !any(flat) {
                return;
            }
            match size {
                LfSize::Four => {}
                LfSize::Six => put::<2>(x, flat, wide::<2, 1, 3>(x)),
                LfSize::Eight => put::<3>(x, flat, wide::<3, 0, 3>(x)),
                LfSize::Sixteen => {
                    let mut rough = splat(0);
                    for k in 4..7 {
                        rough = or(rough, or(gt(abd(x[8 + k], q0), tf), gt(abd(x[7 - k], p0), tf)));
                    }
                    let (flat2, flat1) = (andnot(rough, flat), and(rough, flat));
                    let eight = wide::<3, 0, 3>(x);
                    if any(flat2) {
                        put::<6>(x, flat2, wide::<6, 1, 4>(x));
                    }
                    put::<3>(x, flat1, eight);
                }
            }
        }

        /// The `2 * N` outputs `o` of a wide filter into taps -N .. N where `m` is set.
        #[target_feature(enable = $feature)]
        fn put<const N: usize>(x: &mut [V; 16], m: V, o: [V; 12]) {
            for i in 0..2 * N {
                x[8 - N + i] = sel(m, o[i], x[8 - N + i]);
            }
        }

        /// Wide filter (7.14.6.4) with n = N, the 2-weight taps |j| <= N2 and log2Size = LOG2:
        /// the sliding window of the scalar version, over vectors.
        #[target_feature(enable = $feature)]
        fn wide<const N: usize, const N2: usize, const LOG2: i32>(x: &[V; 16]) -> [V; 12] {
            // e[m] = the tap at offset clamp(m - N, 0, 2N + 1) - (N + 1) from q0.
            let mut e = [splat(0); 25];
            for (m, v) in e[..4 * N + 1].iter_mut().enumerate() {
                *v = x[7 - N + m.saturating_sub(N).min(2 * N + 1)];
            }
            let mut f = [splat(0); 12];
            let mut win = splat(0);
            for v in &e[1..2 * N + 2] {
                win = add(win, *v);
            }
            for i in 0..2 * N {
                if i > 0 {
                    win = add(win, sub(e[i + 1 + 2 * N], e[i]));
                }
                let mut sum = add(win, splat(1 << (LOG2 - 1)));
                for v in &e[i + 1 + N - N2..=i + 1 + N + N2] {
                    sum = add(sum, *v);
                }
                f[i] = srl::<LOG2>(sum);
            }
            f
        }
    };
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(crate) use deblock_kernels;
