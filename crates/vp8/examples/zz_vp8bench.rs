//! Untracked benchmark helper: decode an IVF file VP8_LOOPS times, discarding the pictures.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&args[1]).unwrap();
    let loops: usize = std::env::var("VP8_LOOPS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let hdr_len = usize::from(u16::from_le_bytes([data[6], data[7]]));
    let mut n = 0;
    for _ in 0..loops {
        let mut dec = filmcraft_vp8::Decoder::new();
        let mut pos = hdr_len;
        while pos + 12 <= data.len() {
            let size = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 12;
            let frame = &data[pos..(pos + size).min(data.len())];
            pos += size;
            if let Ok(Some(p)) = dec.decode(frame, 0) {
                n += p.y.len() & 1;
                n += 1;
            }
        }
    }
    eprintln!("{n}");
}
