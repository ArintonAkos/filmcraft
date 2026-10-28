//! Temporary: decode an Annex B file REPS times on THREADS threads, print frames and wall time;
//! with a 4th argument, write the last decode's YUV there. ZZ_AU=1 feeds one access unit per
//! `decode` call (as the MP4 reader does) instead of the whole file.
use std::io::Write;
use std::time::Instant;

fn access_units(data: &[u8]) -> Vec<Vec<u8>> {
    let mut aus: Vec<Vec<u8>> = Vec::new();
    let mut cur = Vec::new();
    let mut has_vcl = false;
    for nal in filmcraft_bitstream::annexb_nals(data) {
        let t = nal[0] & 31;
        let vcl = t == 1 || t == 5;
        let first_mb0 = vcl && nal.len() > 1 && nal[1] & 0x80 != 0;
        if has_vcl && ((vcl && first_mb0) || (6..=9).contains(&t)) {
            aus.push(std::mem::take(&mut cur));
            has_vcl = false;
        }
        cur.extend_from_slice(&[0, 0, 0, 1]);
        cur.extend_from_slice(nal);
        has_vcl |= vcl;
    }
    if !cur.is_empty() {
        aus.push(cur);
    }
    aus
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let threads: usize = a[2].parse().unwrap();
    let reps: usize = a[3].parse().unwrap();
    let aus = if std::env::var_os("ZZ_AU").is_some() { access_units(&data) } else { vec![data.clone()] };
    let mut best = f64::MAX;
    let mut last = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        let mut dec = filmcraft_h264::Decoder::with_threads(threads);
        last.clear();
        for au in &aus {
            last.extend(dec.decode(au, 0).unwrap());
        }
        last.extend(dec.flush());
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("{} frames, best of {reps}: {best:.3} s {:.1} fps (threads {threads}, {} calls)", last.len(), last.len() as f64 / best, aus.len());
    if let Some(p) = a.get(4) {
        let mut o = std::io::BufWriter::new(std::fs::File::create(p).unwrap());
        for p in &last {
            o.write_all(&p.y).unwrap();
            o.write_all(&p.u).unwrap();
            o.write_all(&p.v).unwrap();
        }
    }
}
