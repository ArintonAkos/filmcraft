//! Temporary: decode an IVF file REPS times on THREADS threads, print frames and wall time.
//! With a 4th argument, write the pictures of the first pass there as raw planar YUV.
use std::io::Write;
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let threads: usize = a[2].parse().unwrap();
    let reps: usize = a[3].parse().unwrap();
    let mut frames = Vec::new();
    let mut o = 32;
    while o + 12 <= data.len() {
        let n = u32::from_le_bytes(data[o..o + 4].try_into().unwrap()) as usize;
        frames.push(&data[o + 12..o + 12 + n]);
        o += 12 + n;
    }
    let t = Instant::now();
    let mut shown = 0;
    let mut out = a.get(4).map(|p| std::io::BufWriter::new(std::fs::File::create(p).unwrap()));
    let mut write = |pics: Vec<filmcraft_av1::Picture>| {
        let n = pics.len();
        if let Some(o) = out.as_mut() {
            for p in pics {
                for pl in &p.planes {
                    if p.bit_depth == 8 {
                        o.write_all(&pl.iter().map(|&v| v as u8).collect::<Vec<u8>>()).unwrap();
                    } else {
                        o.write_all(&pl.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
                    }
                }
            }
        }
        n
    };
    for rep in 0..reps {
        let mut dec = filmcraft_av1::Decoder::with_threads(threads);
        for f in &frames {
            let pics = dec.decode(f).unwrap();
            shown += if rep == 0 { write(pics) } else { pics.len() };
        }
        let pics = dec.flush();
        shown += if rep == 0 { write(pics) } else { pics.len() };
        if rep == 0 {
            let st = dec.stats();
            let n = st.frames.max(1) as f64;
            let line: Vec<String> = filmcraft_av1::Stage::ALL.iter().map(|&s| format!("{} {:.2}", s.name(), 1e3 * st.secs(s) / n)).collect();
            eprintln!("ms/frame: {}", line.join("  "));
        }
    }
    let s = t.elapsed().as_secs_f64();
    println!("{shown} frames {s:.3} s {:.1} fps", shown as f64 / s);
}
