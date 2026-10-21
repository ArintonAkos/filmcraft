//! Block inter prediction process (8.5.2.4): separable 8-tap sub-sample interpolation with edge
//! clamping, for unscaled (step 16; the filters are `filmcraft_simd::vp9_predict`) and scaled
//! references.

use crate::frame::Frame;
use crate::tables::SUBPEL_FILTERS;
pub use filmcraft_simd::Vp9Out;

/// Filter taps of `filter` (0..3) at sub-sample position `frac` (0..15).
#[inline(always)]
fn taps(filter: u8, frac: usize) -> &'static [i16; 8] {
    const NONE: [i16; 8] = [0, 0, 0, 128, 0, 0, 0, 0];
    let o = (filter as usize * 16 + frac) * 8;
    SUBPEL_FILTERS.get(o..).and_then(|f| f.first_chunk::<8>()).unwrap_or(&NONE)
}

/// Plane `plane` of a reference frame, valid for x <= last_x, y <= last_y. Rows are read through
/// [`Frame::row`] / [`Frame::span`], which wait until the reference has published them.
pub struct RefPlane<'a> {
    pub frame: &'a Frame,
    pub plane: usize,
    pub last_x: i32,
    pub last_y: i32,
}

impl RefPlane<'_> {
    #[inline]
    fn row(&self, y: usize) -> &[u16] {
        self.frame.row(self.plane, y)
    }
}

/// Predict a `w` x `h` block (`w` 4, 8, 16, 32 or 64) whose top-left sample is at (`x`, `y`) in
/// 1/16 sample units of an unscaled reference, written into `dst` (rows `ds` apart) as `out`
/// says. `win` must hold at least `(w + 7) * (h + 7) + 1` entries (edge-clamped source window,
/// plus the one sample the vector loads read past it).
#[allow(clippy::too_many_arguments)]
pub fn predict_unscaled(r: &RefPlane, x: i32, y: i32, w: usize, h: usize, filter: u8, bit_depth: u8, dst: &mut [u16], ds: usize, out: Vp9Out, win: &mut [u16]) {
    let (x0, y0) = (x >> 4, y >> 4);
    let (fx, fy) = ((x & 15) as usize, (y & 15) as usize);
    let inside = x0 - 3 >= 0 && y0 - 3 >= 0 && x0 + w as i32 + 4 <= r.last_x && y0 + h as i32 + 4 <= r.last_y;
    // Source view with origin at (x0 - 3, y0 - 3): the reference band itself when the window
    // lies inside the picture and inside one band.
    let span = if inside { r.frame.span(r.plane, (y0 - 3) as usize, (y0 + h as i32 + 5) as usize) } else { None };
    let (src, ss): (&[u16], usize) = match span {
        Some((data, o)) => (data.get(o + (x0 - 3) as usize..).unwrap_or_default(), r.frame.strides[r.plane]),
        None => {
            let ws = fetch_window(r, x0, y0, w, h, win);
            (&win[..], ws)
        }
    };
    let (fh, fv) = ((fx != 0).then(|| taps(filter, fx)), (fy != 0).then(|| taps(filter, fy)));
    filmcraft_simd::vp9_predict(src, ss, fh, fv, w, h, ((1u32 << bit_depth) - 1) as u16, dst, ds, out);
}

/// `dst[..n] = src[..n]` with a fixed-length copy for the block widths (and window widths,
/// + 7) that occur, so that each row is a few inline loads and stores rather than a memcpy call.
#[inline(always)]
pub(crate) fn copy_samples(dst: &mut [u16], src: &[u16], n: usize) {
    #[inline(always)]
    fn fixed<const N: usize>(dst: &mut [u16], src: &[u16]) {
        if let (Some(d), Some(s)) = (dst.first_chunk_mut::<N>(), src.first_chunk::<N>()) {
            *d = *s;
        }
    }
    match n {
        4 => fixed::<4>(dst, src),
        8 => fixed::<8>(dst, src),
        16 => fixed::<16>(dst, src),
        32 => fixed::<32>(dst, src),
        64 => fixed::<64>(dst, src),
        11 => fixed::<11>(dst, src),
        15 => fixed::<15>(dst, src),
        23 => fixed::<23>(dst, src),
        39 => fixed::<39>(dst, src),
        71 => fixed::<71>(dst, src),
        _ => copy_samples_any(dst, src, n),
    }
}

/// [`copy_samples`] of any length, out of line: inlined, the compiler merges the fixed-length
/// copies with this variable-length one into one memcpy call of a run-time length.
#[cold]
#[inline(never)]
fn copy_samples_any(dst: &mut [u16], src: &[u16], n: usize) {
    if let (Some(d), Some(s)) = (dst.get_mut(..n), src.get(..n)) {
        d.copy_from_slice(s);
    }
}

/// Copy the (w + 7) x (h + 7) source window starting at (x0 - 3, y0 - 3) with edge clamping.
fn fetch_window(r: &RefPlane, x0: i32, y0: i32, w: usize, h: usize, win: &mut [u16]) -> usize {
    let ws = w + 7;
    for row in 0..h + 7 {
        let yy = (y0 - 3 + row as i32).clamp(0, r.last_y) as usize;
        let src = r.row(yy);
        let dst = &mut win[row * ws..row * ws + ws];
        let xs = x0 - 3;
        if xs >= 0 && xs + ws as i32 - 1 <= r.last_x {
            copy_samples(dst, &src[xs as usize..xs as usize + ws], ws);
        } else {
            for (c, d) in dst.iter_mut().enumerate() {
                *d = src[(xs + c as i32).clamp(0, r.last_x) as usize];
            }
        }
    }
    ws
}

/// General process of 8.5.2.4 (any step): the `w` x `h` block at (`x`, `y`) in 1/16 sample units,
/// stepping `step_x` / `step_y`, into `out` (stride `w`). `tmp` must hold at least
/// `(h * step_y / 16 + 8) * w` entries.
#[allow(clippy::too_many_arguments)]
pub fn predict_scaled(r: &RefPlane, x: i32, y: i32, step_x: i32, step_y: i32, w: usize, h: usize, filter: u8, bit_depth: u8, out: &mut [u16], tmp: &mut [u16]) {
    let max = (1i32 << bit_depth) - 1;
    let ih = ((((h as i32 - 1) * step_y + 15) >> 4) + 8) as usize;
    for row in 0..ih {
        let yy = ((y >> 4) + row as i32 - 3).clamp(0, r.last_y) as usize;
        let src = r.row(yy);
        for c in 0..w {
            let p = x + step_x * c as i32;
            let f = taps(filter, (p & 15) as usize);
            let mut acc = 0i32;
            for t in 0..8 {
                acc += f[t] as i32 * src[((p >> 4) + t as i32 - 3).clamp(0, r.last_x) as usize] as i32;
            }
            tmp[row * w + c] = ((acc + 64) >> 7).clamp(0, max) as u16;
        }
    }
    for rr in 0..h {
        let p = (y & 15) + step_y * rr as i32;
        let f = taps(filter, (p & 15) as usize);
        let base = (p >> 4) as usize;
        for c in 0..w {
            let mut acc = 0i32;
            for t in 0..8 {
                acc += f[t] as i32 * tmp[(base + t) * w + c] as i32;
            }
            out[rr * w + c] = ((acc + 64) >> 7).clamp(0, max) as u16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_path_with_unit_step_matches_unscaled() {
        let (w, h) = (40usize, 30usize);
        let data: Vec<u16> = (0..w * h).map(|i| ((i * 37 + (i / w) * 11) % 256) as u16).collect();
        let info = crate::frame::FrameInfo {
            width: w as u32,
            height: h as u32,
            ss_x: true,
            ss_y: true,
            bit_depth: 8,
            color_space: 0,
            color_range: false,
            render_width: w as u32,
            render_height: h as u32,
            key: true,
            intra_only: false,
        };
        // Planes are allocated in whole superblocks: copy the test picture into a 64x64 plane.
        let mut full = vec![0u16; 64 * 64];
        for y in 0..h {
            full[y * 64..y * 64 + w].copy_from_slice(&data[y * w..y * w + w]);
        }
        let chroma = vec![0u16; 32 * 32];
        let frame = Frame::from_planes(info, [&full, &chroma, &chroma]);
        let r = RefPlane { frame: &frame, plane: 0, last_x: w as i32 - 1, last_y: h as i32 - 1 };
        let mut tmp = vec![0u16; 80 * 80];
        for filter in 0..4u8 {
            for &(x, y) in &[(0, 0), (5, 3), (16 * 7 + 5, 16 * 9 + 11), (-40, -3), (16 * 36 + 1, 16 * 25 + 15), (16 * 10, 16 * 10 + 8)] {
                for &(bw, bh) in &[(4usize, 4usize), (8, 8), (16, 8)] {
                    let mut a = vec![0u16; bw * bh];
                    let mut b = vec![0u16; bw * bh];
                    predict_unscaled(&r, x, y, bw, bh, filter, 8, &mut a, bw, Vp9Out::Put, &mut [0u16; 71 * 71 + 1]);
                    predict_scaled(&r, x, y, 16, 16, bw, bh, filter, 8, &mut b, &mut tmp);
                    assert_eq!(a, b, "filter {filter} pos ({x},{y}) {bw}x{bh}");
                }
            }
        }
    }
}
