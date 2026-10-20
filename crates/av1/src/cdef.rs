//! CDEF (7.15).

use crate::frame::{FrameBuf, RowBand};
use crate::par::Pool;
use crate::postfilter::bands;
use crate::spec_tables::*;
use crate::state::FrameState;

/// Run CDEF on `fs.cur`; returns the CdefFrame, or None when no block is filtered (CdefFrame
/// equals CurrFrame). Bands of 64 luma rows (whole 8x8 blocks) are filtered in parallel: every
/// block reads only CurrFrame and writes only its own samples.
pub(crate) fn apply(fs: &FrameState, pool: &Pool) -> Option<FrameBuf> {
    let fh = &fs.fh;
    if fh.coded_lossless || fh.allow_intrabc || !fs.seq.enable_cdef {
        return None;
    }
    let (mi_rows, mi_cols) = (fh.mi_rows as usize, fh.mi_cols as usize);
    let mut out = fs.cur.clone();
    let ranges = bands(pool, mi_rows * 4);
    let mut planes = out.row_bands(&ranges);
    let any = pool.map_mut(&mut planes, |b, band| {
        let (start, end) = ranges[b];
        let mut any = false;
        let mut r = start / 4;
        while r < mi_rows.min(end / 4) {
            let mut c = 0;
            while c < mi_cols {
                let idx = fs.cdef_idx(r & !15, c & !15);
                if idx >= 0 && cdef_block(fs, band, r, c, idx as usize) {
                    any = true;
                }
                c += 2;
            }
            r += 2;
        }
        any
    });
    drop(planes);
    any.contains(&true).then_some(out)
}

fn cdef_block(fs: &FrameState, out: &mut [RowBand; 3], r: usize, c: usize, idx: usize) -> bool {
    let mi = &fs.mi;
    let sk = |rr: usize, cc: usize| -> bool { if rr < mi.rows && cc < mi.cols { mi.skip[mi.idx(rr, cc)] } else { true } };
    let skip = sk(r, c) && sk(r + 1, c) && sk(r, c + 1) && sk(r + 1, c + 1);
    if skip {
        return false;
    }
    let coeff_shift = fs.bit_depth - 8;
    let (y_dir, var) = direction(fs, r, c);
    let cp = &fs.fh.cdef;
    let mut pri = (cp.y_pri[idx] << coeff_shift) as i32;
    let sec = (cp.y_sec[idx] << coeff_shift) as i32;
    let dir = if pri == 0 { 0 } else { y_dir };
    let var_str = if (var >> 6) != 0 { (31 - ((var >> 6) as u32).leading_zeros()).min(12) as i32 } else { 0 };
    pri = if var != 0 { (pri * (4 + var_str) + 8) >> 4 } else { 0 };
    let damping = cp.damping as i32 + coeff_shift as i32;
    filter(fs, out, 0, r, c, pri, sec, damping, dir);
    if fs.num_planes == 1 {
        return true;
    }
    let pri = (cp.uv_pri[idx] << coeff_shift) as i32;
    let sec = (cp.uv_sec[idx] << coeff_shift) as i32;
    let dir = if pri == 0 { 0 } else { CDEF_UV_DIR[fs.ssx][fs.ssy][y_dir] as usize };
    let damping = cp.damping as i32 + coeff_shift as i32 - 1;
    filter(fs, out, 1, r, c, pri, sec, damping, dir);
    filter(fs, out, 2, r, c, pri, sec, damping, dir);
    true
}

fn direction(fs: &FrameState, r: usize, c: usize) -> (usize, i32) {
    let x0 = c << 2;
    let y0 = r << 2;
    let pl = &fs.cur.planes[0];
    let rows: [&[u16; 8]; 8] = std::array::from_fn(|i| {
        let row = &pl.row(y0 + i)[x0..x0 + 8];
        row.first_chunk::<8>().unwrap_or(&[0; 8])
    });
    filmcraft_simd::cdef_direction(&rows, fs.bit_depth - 8)
}

/// Marks a sample outside the frame (CdefAvailable = 0) in the padded block.
const CDEF_NA: i32 = i32::MIN;
/// Padded block: 8x8 plus 2 samples on every side, row stride 12.
const PAD_STRIDE: usize = 12;

#[inline(always)]
fn constrain(diff: i32, threshold: i32, adj: u32) -> i32 {
    let mag = (threshold - (diff.abs() >> adj)).clamp(0, diff.abs());
    if diff < 0 { -mag } else { mag }
}

#[allow(clippy::too_many_arguments)]
fn filter(fs: &FrameState, out: &mut [RowBand; 3], plane: usize, r: usize, c: usize, pri: i32, sec: i32, damping: i32, dir: usize) {
    if pri == 0 && sec == 0 {
        // Every tap contributes 0 and the clamp to [min, max] keeps x: CdefFrame = CurrFrame.
        return;
    }
    let coeff_shift = fs.bit_depth - 8;
    let (sub_x, sub_y) = if plane > 0 { (fs.ssx, fs.ssy) } else { (0, 0) };
    let x0 = (c * 4) >> sub_x;
    let y0 = (r * 4) >> sub_y;
    let w = 8 >> sub_x;
    let h = 8 >> sub_y;
    let src = &fs.cur.planes[plane];
    let (mi_rows, mi_cols) = (fs.fh.mi_rows as isize, fs.fh.mi_cols as isize);
    let pri_tap = ((pri >> coeff_shift) & 1) as usize;
    let off = |d: usize, k: usize| CDEF_DIRECTIONS[d][k][0] as isize * PAD_STRIDE as isize + CDEF_DIRECTIONS[d][k][1] as isize;
    let pri_off = [off(dir, 0), off(dir, 1)];
    let d_lo = (dir + 6) & 7;
    let d_hi = (dir + 2) & 7;
    let sec_off = [[off(d_lo, 0), off(d_hi, 0)], [off(d_lo, 1), off(d_hi, 1)]];
    let pri_taps = [CDEF_PRI_TAPS[pri_tap][0] as i32, CDEF_PRI_TAPS[pri_tap][1] as i32];
    let sec_taps = [CDEF_SEC_TAPS[pri_tap][0] as i32, CDEF_SEC_TAPS[pri_tap][1] as i32];
    let adj = |t: i32| if t == 0 { 0 } else { (damping - (31 - (t as u32).leading_zeros()) as i32).max(0) as u32 };
    let (pri_adj, sec_adj) = (adj(pri), adj(sec));
    let dst = &mut out[plane];
    // CdefAvailable per sample: its 4x4 block is inside the frame. The padded block lies inside
    // when its first and last rows and columns do.
    let inside = |v: isize, sub: usize, mi: isize| v >= 0 && (v << sub) >> 2 < mi;
    let (top, left) = (y0 as isize - 2, x0 as isize - 2);
    if inside(top, sub_y, mi_rows)
        && inside(top + h as isize + 3, sub_y, mi_rows)
        && inside(left, sub_x, mi_cols)
        && inside(left + w as isize + 3, sub_x, mi_cols)
    {
        // Every tap is available: whole rows at a time (constrain() with threshold 0 is 0).
        let mut buf = [0i16; PAD_STRIDE * PAD_STRIDE];
        for (i, d) in buf.as_chunks_mut::<PAD_STRIDE>().0.iter_mut().take(h + 4).enumerate() {
            for (d, &s) in d.iter_mut().zip(&src.row(y0 + i - 2)[x0 - 2..x0 + w + 2]) {
                *d = s as i16;
            }
        }
        let t = Taps { pri_off, sec_off, pri_taps, sec_taps, pri, sec, pri_adj, sec_adj };
        filter_inside(&buf, &t, dst, (y0, x0, h), w);
        return;
    }
    let mut buf = [CDEF_NA; PAD_STRIDE * PAD_STRIDE];
    for i in 0..h + 4 {
        let y = top + i as isize;
        if !inside(y, sub_y, mi_rows) {
            continue;
        }
        let row = src.row(y as usize);
        for j in 0..w + 4 {
            let x = left + j as isize;
            if inside(x, sub_x, mi_cols) {
                buf[i * PAD_STRIDE + j] = row[x as usize] as i32;
            }
        }
    }
    for i in 0..h {
        // The centre sample is read even where CdefAvailable is 0 (8x8 blocks straddling an odd
        // MiRows / MiCols edge).
        let srow = &src.row(y0 + i)[x0..x0 + w];
        let drow = &mut dst.row_mut(y0 + i)[x0..x0 + w];
        for j in 0..w {
            let ci = ((i + 2) * PAD_STRIDE + j + 2) as isize;
            let x = srow[j] as i32;
            let mut sum = 0i32;
            let mut max = x;
            let mut min = x;
            for k in 0..2 {
                for sign in [-1isize, 1] {
                    let p = buf[(ci + sign * pri_off[k]) as usize];
                    if p != CDEF_NA {
                        if pri != 0 {
                            sum += pri_taps[k] * constrain(p - x, pri, pri_adj);
                        }
                        max = max.max(p);
                        min = min.min(p);
                    }
                    for s in 0..2 {
                        let q = buf[(ci + sign * sec_off[k][s]) as usize];
                        if q != CDEF_NA {
                            if sec != 0 {
                                sum += sec_taps[k] * constrain(q - x, sec, sec_adj);
                            }
                            max = max.max(q);
                            min = min.min(q);
                        }
                    }
                }
            }
            drow[j] = (x + ((8 + sum - (sum < 0) as i32) >> 4)).clamp(min, max) as u16;
        }
    }
}

/// A block's filter taps: offsets in the padded block, weights, strengths and damping shifts.
struct Taps {
    pri_off: [isize; 2],
    sec_off: [[isize; 2]; 2],
    pri_taps: [i32; 2],
    sec_taps: [i32; 2],
    pri: i32,
    sec: i32,
    pri_adj: u32,
    sec_adj: u32,
}

/// CDEF of a `w` x `h` block whose taps are all available, from its padded samples `buf`.
fn filter_inside(buf: &[i16; PAD_STRIDE * PAD_STRIDE], t: &Taps, dst: &mut RowBand, rows: (usize, usize, usize), w: usize) {
    // Taps of a zero strength add 0. With one strength the tap weights sum to 12 < 16, so the
    // result stays between the centre and its farthest tap and the clamp to the taps' range
    // changes nothing (libaom and dav1d skip it too).
    let mut taps = [(0isize, 0i32, 0i32, 0u32); 12];
    let mut n = 0;
    for k in 0..2 {
        for sign in [-1isize, 1] {
            if t.pri != 0 {
                taps[n] = (sign * t.pri_off[k], t.pri_taps[k], t.pri, t.pri_adj);
                n += 1;
            }
            if t.sec != 0 {
                taps[n] = (sign * t.sec_off[k][0], t.sec_taps[k], t.sec, t.sec_adj);
                taps[n + 1] = (sign * t.sec_off[k][1], t.sec_taps[k], t.sec, t.sec_adj);
                n += 2;
            }
        }
    }
    let taps = &taps[..n];
    match (w, t.pri != 0 && t.sec != 0) {
        (8, true) => filter_rows::<8, true>(buf, taps, dst, rows),
        (8, false) => filter_rows::<8, false>(buf, taps, dst, rows),
        (_, true) => filter_rows::<4, true>(buf, taps, dst, rows),
        (_, false) => filter_rows::<4, false>(buf, taps, dst, rows),
    }
}

/// CDEF of a block whose taps are all available, `W` samples per row, clamped to the taps'
/// range when `CLAMP`. Samples (at most 12 bits), differences and the weighted tap sum all
/// fit in i16, which doubles the SIMD width.
#[inline(always)]
fn filter_rows<const W: usize, const CLAMP: bool>(
    b16: &[i16; PAD_STRIDE * PAD_STRIDE],
    taps: &[(isize, i32, i32, u32)],
    dst: &mut RowBand,
    (y0, x0, h): (usize, usize, usize),
) {
    for i in 0..h {
        let ci = ((i + 2) * PAD_STRIDE + 2) as isize;
        let Some(&x) = b16.get(ci as usize..).and_then(|r| r.first_chunk::<W>()) else {
            return;
        };
        let (mut sum, mut max, mut min) = ([0i16; W], x, x);
        for &(off, weight, thr, adj) in taps {
            let (weight, thr) = (weight as i16, thr as i16);
            let start = (ci + off) as usize;
            let Some(p) = b16.get(start..).and_then(|r| r.first_chunk::<W>()) else {
                return;
            };
            for j in 0..W {
                // constrain(): .max / .min instead of clamp (whose bound check panics), and
                // the sign applied branch-free, so the loop vectorises.
                let d = p[j] - x[j];
                let ad = d.abs();
                let mag = (thr - (ad >> adj)).max(0).min(ad);
                let sign = d >> 15;
                sum[j] += weight * ((mag ^ sign) - sign);
                if CLAMP {
                    max[j] = max[j].max(p[j]);
                    min[j] = min[j].min(p[j]);
                }
            }
        }
        let drow = &mut dst.row_mut(y0 + i)[x0..x0 + W];
        for j in 0..W {
            let s = sum[j];
            let v = x[j] + ((8 + s - (s < 0) as i16) >> 4);
            drow[j] = if CLAMP { v.max(min[j]).min(max[j]) } else { v } as u16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Plane;

    /// 7.15.3 as the spec writes it: all twelve taps, then the clamp to their range.
    fn spec_filter(buf: &[i16; PAD_STRIDE * PAD_STRIDE], t: &Taps, w: usize, h: usize) -> Vec<u16> {
        let mut out = Vec::new();
        for i in 0..h {
            for j in 0..w {
                let ci = ((i + 2) * PAD_STRIDE + j + 2) as isize;
                let x = buf[ci as usize] as i32;
                let (mut sum, mut max, mut min) = (0, x, x);
                for k in 0..2 {
                    for sign in [-1isize, 1] {
                        let p = buf[(ci + sign * t.pri_off[k]) as usize] as i32;
                        sum += t.pri_taps[k] * constrain(p - x, t.pri, t.pri_adj);
                        (max, min) = (max.max(p), min.min(p));
                        for s in 0..2 {
                            let q = buf[(ci + sign * t.sec_off[k][s]) as usize] as i32;
                            sum += t.sec_taps[k] * constrain(q - x, t.sec, t.sec_adj);
                            (max, min) = (max.max(q), min.min(q));
                        }
                    }
                }
                out.push((x + ((8 + sum - (sum < 0) as i32) >> 4)).clamp(min, max) as u16);
            }
        }
        out
    }

    /// The vector path, without the zero-strength taps and (with one strength) the clamp,
    /// gives the spec's samples: random and edge-like blocks, every strength, damping and
    /// direction, at 8, 10 and 12 bits.
    #[test]
    fn fast_path_matches_the_spec_filter() {
        let mut seed = 0x1234_5679u32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let off = |d: usize, k: usize| CDEF_DIRECTIONS[d][k][0] as isize * PAD_STRIDE as isize + CDEF_DIRECTIONS[d][k][1] as isize;
        let mut clamped = 0;
        for case in 0..30_000 {
            let shift = [0, 2, 4][case % 3];
            let max = (256i32 << shift) - 1;
            let (w, h) = [(8, 8), (4, 4), (4, 8)][next() as usize % 3];
            let base = (next() % (max as u32 + 1)) as i32;
            let spread = [2, 8, 40, 300, 4096][next() as usize % 5].min(max + 1);
            let buf: [i16; PAD_STRIDE * PAD_STRIDE] = std::array::from_fn(|_| (base + (next() % spread as u32) as i32 - spread / 2).clamp(0, max) as i16);
            let pri = [0, 1, 2, 3, 5, 8, 15][next() as usize % 7] << shift;
            let sec = [0, 1, 2, 4][next() as usize % 4] << shift;
            if pri == 0 && sec == 0 {
                continue;
            }
            let damping = 3 + (next() % 4) as i32 + shift;
            let dir = next() as usize % 8;
            let pri_tap = ((pri >> shift) & 1) as usize;
            let adj = |t: i32| if t == 0 { 0 } else { (damping - (31 - (t as u32).leading_zeros()) as i32).max(0) as u32 };
            let (d_lo, d_hi) = ((dir + 6) & 7, (dir + 2) & 7);
            let t = Taps {
                pri_off: [off(dir, 0), off(dir, 1)],
                sec_off: [[off(d_lo, 0), off(d_hi, 0)], [off(d_lo, 1), off(d_hi, 1)]],
                pri_taps: [CDEF_PRI_TAPS[pri_tap][0] as i32, CDEF_PRI_TAPS[pri_tap][1] as i32],
                sec_taps: [CDEF_SEC_TAPS[pri_tap][0] as i32, CDEF_SEC_TAPS[pri_tap][1] as i32],
                pri,
                sec,
                pri_adj: adj(pri),
                sec_adj: adj(sec),
            };
            let want = spec_filter(&buf, &t, w, h);
            let mut plane = Plane::new(w, h);
            let mut bands = plane.row_bands(&[(0, h)]);
            let Some(band) = bands.first_mut() else { panic!("one band") };
            filter_inside(&buf, &t, band, (0, 0, h), w);
            drop(bands);
            let got: Vec<u16> = (0..h).flat_map(|i| plane.row(i)[..w].to_vec()).collect();
            assert_eq!(got, want, "case {case}: {w}x{h} pri {pri} sec {sec} damping {damping} dir {dir}");
            clamped += (pri != 0 && sec != 0) as u32;
        }
        assert!(clamped > 5000);
    }
}
