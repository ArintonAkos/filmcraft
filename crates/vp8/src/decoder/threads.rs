//! The macroblock rows of a frame on worker threads (the `threads` feature), a pipeline of row
//! stages:
//!
//! - reading: the prediction records (first partition) and the tokens (9.5). From 4 threads
//!   the first partition is read on a thread of its own and the token partitions on up to
//!   `threads - 3` others; row `y` reads partition `y % n` once the row above has read the
//!   tokens of the macroblock above (the non-zero contexts, 13.3), a wavefront one macroblock
//!   deep.
//! - reconstruction. From 3 threads it writes each row into a buffer of its own, under a copy
//!   of the unfiltered last line of the row above (what intra prediction reads, 12), and the
//!   loop filter (15) runs on a thread of its own as the rows arrive; with 2, the loop filter
//!   follows the reconstruction one row behind, as on the calling thread.

use std::sync::atomic::{AtomicBool, AtomicU16, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, PoisonError};

use super::{B_DC_PRED, Decoder, Frame, FrameParams, MbRec, MbResidual, ModeCtx, loop_filter_row, read_mb_tokens, reconstruct_mb};
use crate::boolcoder::BoolDecoder;
use crate::error::{Error, Result};

/// The prediction records of one macroblock row and the coefficients of its macroblocks.
struct RowWork {
    recs: Vec<MbRec>,
    res: Vec<MbResidual>,
}

/// A reconstructed row on its way to the loop filter: the two-row buffer it is in (from luma
/// row `first`: 0 for the frame's first row, 16 for the others, under the line above) and the
/// loop filter level and inner edge flag of each macroblock.
struct Reconstructed {
    buf: Frame,
    first: usize,
    lf: Vec<(u8, bool)>,
}

/// Rows handed from one stage to the next, by row index: [`Rows::take`] waits for its row.
struct Rows<T> {
    /// The rows not yet taken, and whether the frame was abandoned.
    slots: Mutex<(Vec<Option<T>>, bool)>,
    ready: Condvar,
}

impl<T> Rows<T> {
    fn new(n: usize) -> Rows<T> {
        Rows { slots: Mutex::new(((0..n).map(|_| None).collect(), false)), ready: Condvar::new() }
    }

    fn put(&self, y: usize, row: T) {
        let mut s = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(slot) = s.0.get_mut(y) {
            *slot = Some(row);
        }
        self.ready.notify_all();
    }

    /// Row `y` once it is there; None when the frame was abandoned first.
    fn take(&self, y: usize) -> Option<T> {
        let mut s = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            if let Some(row) = s.0.get_mut(y).and_then(Option::take) {
                return Some(row);
            }
            if s.1 {
                return None;
            }
            s = self.ready.wait(s).unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn abandon(&self) {
        self.slots.lock().unwrap_or_else(PoisonError::into_inner).1 = true;
        self.ready.notify_all();
    }
}

/// What the stages of one frame share.
struct Shared {
    mb_w: usize,
    /// Prediction records, from the first partition reader to the token readers.
    headers: Rows<Vec<MbRec>>,
    /// Rows read, to the reconstruction.
    rows: Rows<RowWork>,
    /// Rows reconstructed, to the loop filter (when it runs on a thread of its own), and the
    /// buffers it is done with.
    reconstructed: Rows<Reconstructed>,
    spare_bufs: Mutex<Vec<Frame>>,
    /// Coefficient rows the reconstruction is done with (cleared), for the readers to reuse.
    spare: Mutex<Vec<Vec<MbResidual>>>,
    failed: AtomicBool,
    /// Token readers of separate partitions: macroblocks of each row whose tokens are read, and
    /// the non-zero contexts above each macroblock column (bit i: context i).
    read: Vec<Padded<AtomicUsize>>,
    above_nz: Vec<AtomicU16>,
}

/// A value on a cache line of its own (128 bytes: the line of Apple cores, two of most others),
/// so the readers of neighbouring rows do not invalidate each other's counters.
#[repr(align(128))]
struct Padded<T>(T);

impl Shared {
    fn new(mb_w: usize, mb_h: usize) -> Shared {
        Shared {
            mb_w,
            headers: Rows::new(mb_h),
            rows: Rows::new(mb_h),
            reconstructed: Rows::new(mb_h),
            spare_bufs: Mutex::new(Vec::new()),
            spare: Mutex::new(Vec::new()),
            failed: AtomicBool::new(false),
            read: (0..mb_h).map(|_| Padded(AtomicUsize::new(0))).collect(),
            above_nz: (0..mb_w).map(|_| AtomicU16::new(0)).collect(),
        }
    }

    /// Stops every stage: the waiting ones return.
    fn abandon(&self) {
        self.failed.store(true, Ordering::Relaxed);
        self.headers.abandon();
        self.rows.abandon();
        self.reconstructed.abandon();
    }

    fn abandoned(&self) -> bool {
        self.failed.load(Ordering::Relaxed)
    }

    /// A row of zero coefficients: one the reconstruction cleared, or a new one.
    fn spare_row(&self) -> Vec<MbResidual> {
        let row = self.spare.lock().unwrap_or_else(PoisonError::into_inner).pop();
        row.unwrap_or_else(|| (0..self.mb_w).map(|_| MbResidual::default()).collect())
    }

    /// Waits until row `y` has read the tokens of at least `n` macroblocks and returns how many
    /// it has; None when abandoned.
    fn wait_read(&self, y: usize, n: usize) -> Option<usize> {
        let read = &self.read.get(y)?.0;
        let mut spins = 0u32;
        loop {
            let done = read.load(Ordering::Acquire);
            if done >= n {
                return Some(done);
            }
            if self.abandoned() {
                return None;
            }
            spins += 1;
            if spins < 100 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

fn pack_nz(nz: &[u8; 9]) -> u16 {
    nz.iter().enumerate().fold(0, |m, (i, &v)| m | u16::from(v & 1) << i)
}

fn unpack_nz(m: u16) -> [u8; 9] {
    std::array::from_fn(|i| (m >> i & 1) as u8)
}

/// [`Decoder::decode_rows`] on `pool` (at least 2 threads), its stages as the module says.
pub(super) fn decode_rows(
    dec: &Decoder,
    pool: &rayon::ThreadPool,
    threads: usize,
    bd: &mut BoolDecoder,
    parts: Vec<BoolDecoder>,
    f: &FrameParams,
    seg_map: &mut [u8],
) -> Result<Frame> {
    let n = parts.len();
    let readers = if threads >= 4 { n.min(threads - 3) } else { 1 };
    let sh = Shared::new(dec.mb_w, dec.mb_h);
    let (mut out, mut recon_out) = (None, Ok(()));
    pool.scope(|s| {
        let (sh, out, recon_out) = (&sh, &mut out, &mut recon_out);
        if threads >= 3 {
            s.spawn(move |_| {
                *recon_out = reconstruct_rows(dec, f, sh);
                if recon_out.is_err() {
                    sh.abandon();
                }
            });
            s.spawn(move |_| {
                let r = filter_rows(dec, f, sh);
                if r.is_err() {
                    sh.abandon();
                }
                *out = Some(r);
            });
        } else {
            s.spawn(move |_| {
                let r = reconstruct(dec, f, sh);
                if r.is_err() {
                    sh.abandon();
                }
                *out = Some(r);
            });
        }
        if threads < 4 {
            read_rows(dec, bd, parts, f, seg_map, sh);
            return;
        }
        let mut groups: Vec<Vec<(usize, BoolDecoder)>> = (0..readers).map(|_| Vec::new()).collect();
        for (p, td) in parts.into_iter().enumerate() {
            if let Some(g) = groups.get_mut(p % readers) {
                g.push((p, td));
            }
        }
        for g in groups {
            s.spawn(move |_| read_tokens(dec, f, g, n, sh));
        }
        read_headers(dec, bd, f, seg_map, sh);
    });
    // The reconstruction's error, not the loop filter's "abandoned".
    recon_out?;
    out.unwrap_or_else(|| Err(Error::Invalid("frame reconstruction did not run".into())))
}

/// Prediction records and tokens of every row, in order (any number of partitions).
fn read_rows(dec: &Decoder, bd: &mut BoolDecoder, mut parts: Vec<BoolDecoder>, f: &FrameParams, seg_map: &mut [u8], sh: &Shared) {
    let mut mc = ModeCtx::new(dec.mb_w, dec.mb_h);
    let mut above_nz = vec![[0u8; 9]; dec.mb_w];
    let n = parts.len();
    for mb_y in 0..dec.mb_h {
        let Some(td) = parts.get_mut(mb_y % n).filter(|_| !sh.abandoned()) else {
            sh.abandon();
            return;
        };
        mc.left_bmodes = [B_DC_PRED; 4];
        let mut left_nz = [0u8; 9];
        let mut res = sh.spare_row();
        let mut recs = Vec::with_capacity(dec.mb_w);
        for (mb_x, (an, r)) in above_nz.iter_mut().zip(res.iter_mut()).enumerate() {
            let rec = dec.read_mb_header(bd, f, seg_map, &mut mc, (mb_x, mb_y));
            read_mb_tokens(td, f, &dec.ctx.coeff, &rec, (an, &mut left_nz), r);
            recs.push(rec);
        }
        sh.rows.put(mb_y, RowWork { recs, res });
    }
}

/// Prediction records of every row (the first partition), for the token readers.
fn read_headers(dec: &Decoder, bd: &mut BoolDecoder, f: &FrameParams, seg_map: &mut [u8], sh: &Shared) {
    let mut mc = ModeCtx::new(dec.mb_w, dec.mb_h);
    for mb_y in 0..dec.mb_h {
        if sh.abandoned() {
            return;
        }
        mc.left_bmodes = [B_DC_PRED; 4];
        let recs = (0..dec.mb_w).map(|mb_x| dec.read_mb_header(bd, f, seg_map, &mut mc, (mb_x, mb_y))).collect();
        sh.headers.put(mb_y, recs);
    }
}

/// Tokens of the rows of the partitions in `group` (of `n`), each macroblock once the row above
/// has read the one above it.
fn read_tokens(dec: &Decoder, f: &FrameParams, mut group: Vec<(usize, BoolDecoder)>, n: usize, sh: &Shared) {
    for mb_y in 0..dec.mb_h {
        let Some((_, td)) = group.iter_mut().find(|(p, _)| *p == mb_y % n) else { continue };
        let Some(recs) = sh.headers.take(mb_y) else { return };
        let mut res = sh.spare_row();
        let mut left_nz = [0u8; 9];
        // Macroblocks of the row above known to be read (all of them above the first row).
        let mut above = if mb_y == 0 { dec.mb_w } else { 0 };
        for (mb_x, ((rec, r), cell)) in recs.iter().zip(res.iter_mut()).zip(&sh.above_nz).enumerate() {
            if above <= mb_x {
                let Some(n) = sh.wait_read(mb_y - 1, mb_x + 1) else { return };
                above = n;
            }
            let mut an = unpack_nz(cell.load(Ordering::Relaxed));
            read_mb_tokens(td, f, &dec.ctx.coeff, rec, (&mut an, &mut left_nz), r);
            cell.store(pack_nz(&an), Ordering::Relaxed);
            if let Some(read) = sh.read.get(mb_y) {
                read.0.store(mb_x + 1, Ordering::Release);
            }
        }
        sh.rows.put(mb_y, RowWork { recs, res });
    }
}

/// Reconstruction of the rows as they are read, the loop filter one row behind.
fn reconstruct(dec: &Decoder, f: &FrameParams, sh: &Shared) -> Result<Frame> {
    let (mb_w, mb_h) = (dec.mb_w, dec.mb_h);
    let mut frame = Frame::new(mb_w * 16, mb_h * 16);
    let mut lf = [vec![(0u8, false); mb_w], vec![(0u8, false); mb_w]];
    let filter = f.hdr.filter_level > 0;
    for mb_y in 0..mb_h {
        let Some(RowWork { recs, mut res }) = sh.rows.take(mb_y) else {
            return Err(Error::Invalid("frame abandoned".into()));
        };
        for (mb_x, ((rec, r), l)) in recs.iter().zip(res.iter_mut()).zip(lf[0].iter_mut()).enumerate() {
            *l = (dec.filter_level(f.hdr, rec.segment, &rec.modes.info), !rec.has_y2() || r.has_coeffs);
            reconstruct_mb(&mut frame, (mb_x, mb_y), 0, mb_w, &rec.modes, r, &f.recon)?;
            r.clear();
        }
        sh.spare.lock().unwrap_or_else(PoisonError::into_inner).push(res);
        if filter && mb_y > 0 {
            loop_filter_row(&mut frame, mb_y - 1, &lf[1], f.hdr, f.key_frame);
        }
        lf.swap(0, 1);
    }
    if filter && mb_h > 0 {
        loop_filter_row(&mut frame, mb_h - 1, &lf[1], f.hdr, f.key_frame);
    }
    Ok(frame)
}

/// Reconstruction of the rows as they are read, each into a buffer of its own for the loop
/// filter thread: the unfiltered last lines of the row above are kept for intra prediction.
fn reconstruct_rows(dec: &Decoder, f: &FrameParams, sh: &Shared) -> Result<()> {
    let (mb_w, mb_h) = (dec.mb_w, dec.mb_h);
    let (w, cw) = (mb_w * 16, mb_w * 8);
    let mut above = (vec![0u8; w], vec![0u8; cw], vec![0u8; cw]);
    for mb_y in 0..mb_h {
        let Some(RowWork { recs, mut res }) = sh.rows.take(mb_y) else {
            return Err(Error::Invalid("frame abandoned".into()));
        };
        let spare = sh.spare_bufs.lock().unwrap_or_else(PoisonError::into_inner).pop();
        let mut buf = spare.unwrap_or_else(|| Frame::new(w, 32));
        // The frame's first row at the buffer's first row (where intra prediction sees the top
        // of the frame), the others at its second, under the line above.
        let first = if mb_y == 0 { 0 } else { 16 };
        if mb_y > 0 {
            copy_lines(&mut buf.y, 15 * w, &above.0, 0, w);
            copy_lines(&mut buf.u, 7 * cw, &above.1, 0, cw);
            copy_lines(&mut buf.v, 7 * cw, &above.2, 0, cw);
        }
        let top = mb_y * 16 - first;
        let mut lf = Vec::with_capacity(mb_w);
        for (mb_x, (rec, r)) in recs.iter().zip(res.iter_mut()).enumerate() {
            lf.push((dec.filter_level(f.hdr, rec.segment, &rec.modes.info), !rec.has_y2() || r.has_coeffs));
            reconstruct_mb(&mut buf, (mb_x, mb_y), top, mb_w, &rec.modes, r, &f.recon)?;
            r.clear();
        }
        sh.spare.lock().unwrap_or_else(PoisonError::into_inner).push(res);
        copy_lines(&mut above.0, 0, &buf.y, (first + 15) * w, w);
        copy_lines(&mut above.1, 0, &buf.u, (first / 2 + 7) * cw, cw);
        copy_lines(&mut above.2, 0, &buf.v, (first / 2 + 7) * cw, cw);
        sh.reconstructed.put(mb_y, Reconstructed { buf, first, lf });
    }
    Ok(())
}

/// The reconstructed rows into the frame, each filtered as it arrives (its intra prediction is
/// done: the rows below read the copies of its unfiltered last lines).
fn filter_rows(dec: &Decoder, f: &FrameParams, sh: &Shared) -> Result<Frame> {
    let (mb_w, mb_h) = (dec.mb_w, dec.mb_h);
    let (w, cw) = (mb_w * 16, mb_w * 8);
    let mut frame = Frame::new(w, mb_h * 16);
    for mb_y in 0..mb_h {
        let Some(Reconstructed { buf, first, lf }) = sh.reconstructed.take(mb_y) else {
            return Err(Error::Invalid("frame abandoned".into()));
        };
        copy_lines(&mut frame.y, mb_y * 16 * w, &buf.y, first * w, 16 * w);
        copy_lines(&mut frame.u, mb_y * 8 * cw, &buf.u, first / 2 * cw, 8 * cw);
        copy_lines(&mut frame.v, mb_y * 8 * cw, &buf.v, first / 2 * cw, 8 * cw);
        sh.spare_bufs.lock().unwrap_or_else(PoisonError::into_inner).push(buf);
        if f.hdr.filter_level > 0 {
            loop_filter_row(&mut frame, mb_y, &lf, f.hdr, f.key_frame);
        }
    }
    Ok(frame)
}

/// `n` samples of `src` from `s` into `dst` from `d` (nothing when either does not hold them).
fn copy_lines(dst: &mut [u8], d: usize, src: &[u8], s: usize, n: usize) {
    if let (Some(dst), Some(src)) = (dst.get_mut(d..d + n), src.get(s..s + n)) {
        dst.copy_from_slice(src);
    }
}

#[cfg(test)]
mod tests {
    use super::Rows;

    /// A stage waiting for a row returns once the frame is abandoned instead of waiting forever.
    #[test]
    fn abandoning_wakes_a_waiting_stage() {
        let rows = Rows::<u8>::new(4);
        std::thread::scope(|s| {
            let waiting = s.spawn(|| rows.take(2));
            rows.put(1, 7);
            rows.abandon();
            assert_eq!(waiting.join().unwrap(), None);
        });
        assert_eq!(rows.take(1), Some(7), "rows put before still come out");
    }
}
