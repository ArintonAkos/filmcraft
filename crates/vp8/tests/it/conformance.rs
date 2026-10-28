//! Bit-exactness against libvpx (through ffmpeg) on generated fixtures covering every coding
//! tool: skipped when ffmpeg or its libvpx encoder is absent.

use crate::common;

fn check(name: &str) {
    let f = common::fixture(name);
    let Some((ivf, yuv)) = common::ensure(f) else { return };
    let data = std::fs::read(&ivf).unwrap();
    let reference = std::fs::read(&yuv).unwrap();
    let pictures = common::decode_all(&common::ivf_frames(&data));
    let (w, h) = (f.width as usize, f.height as usize);
    let size = w * h + 2 * w.div_ceil(2) * h.div_ceil(2);
    assert_eq!(pictures.len(), reference.len() / size, "{name}: picture count");
    for (i, p) in pictures.iter().enumerate() {
        assert_eq!((p.width as usize, p.height as usize), (w, h), "{name}: size of picture {i}");
        assert!(common::i420(p) == reference[i * size..(i + 1) * size], "{name}: picture {i} differs from libvpx");
    }
}

macro_rules! fixture_tests {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                check(stringify!($name));
            }
        )*
    };
}

fixture_tests!(
    default,
    odd_size,
    tiny,
    intra_only,
    noise_high_rate,
    near_lossless,
    coarse,
    partitions8,
    altref_2pass,
    sharpness7,
    version1_bilinear_simple,
    version2,
    version3_fullpixel,
    realtime_segments,
    screen,
    temporal_layers,
    hd_1080p,
);

/// Every thread count decodes to the same pictures: with more than one, the rows of each frame
/// go through a pipeline of threads (and with several token partitions, a wavefront). The last
/// stream changes size at each of its key frames.
#[test]
fn same_pictures_at_every_thread_count() {
    let names = ["default", "odd_size", "tiny", "intra_only", "partitions8", "altref_2pass", "version1_bilinear_simple", "realtime_segments", "hd_1080p"];
    let mut streams = Vec::new();
    for name in names {
        let Some((ivf, _)) = common::ensure(common::fixture(name)) else { return };
        streams.push((name, std::fs::read(&ivf).unwrap()));
    }
    let mut cases: Vec<(&str, Vec<&[u8]>)> = streams.iter().map(|(name, data)| (*name, common::ivf_frames(data))).collect();
    let resized = ["odd_size", "hd_1080p", "tiny", "partitions8"]
        .iter()
        .flat_map(|n| cases.iter().find(|(name, _)| name == n).map(|c| c.1.clone()).unwrap_or_default())
        .collect();
    cases.push(("odd_size, hd_1080p, tiny, partitions8", resized));
    for (name, frames) in cases {
        let decode = |threads: usize| {
            let mut dec = filmcraft_vp8::Decoder::with_threads(threads);
            frames.iter().enumerate().filter_map(|(i, f)| dec.decode(f, i as i64).unwrap()).map(|p| (p.width, common::i420(&p))).collect::<Vec<_>>()
        };
        let one = decode(1);
        assert!(!one.is_empty(), "{name}: pictures");
        for threads in [2, 3, 4, 5, 8] {
            assert!(decode(threads) == one, "{name}: {threads} threads");
        }
    }
}

/// Hidden (altref) frames produce no picture, and leaving out the frames the decoder calls
/// disposable changes none of the other pictures.
#[test]
fn hidden_and_disposable_frames() {
    for name in ["altref_2pass", "temporal_layers"] {
        let f = common::fixture(name);
        let Some((ivf, _)) = common::ensure(f) else { return };
        let data = std::fs::read(&ivf).unwrap();
        let frames = common::ivf_frames(&data);
        // every picture, and which frames could be left out (asked right before each is decoded)
        let mut dec = filmcraft_vp8::Decoder::new();
        let mut all = Vec::new();
        let mut disposable = Vec::new();
        for (i, f) in frames.iter().enumerate() {
            disposable.push(dec.is_disposable(f));
            if let Some(p) = dec.decode(f, i as i64).unwrap() {
                all.push((p.pts, common::i420(&p)));
            }
        }
        let stats = dec.stats();
        assert_eq!(stats.frames as usize, frames.len());
        assert_eq!(all.len() + stats.hidden_frames as usize, frames.len(), "{name}: one picture per shown frame");
        if name == "altref_2pass" {
            assert!(stats.hidden_frames > 0, "{name}: has hidden altref frames");
        }
        if name == "temporal_layers" {
            assert!(disposable.iter().filter(|&&d| d).count() >= frames.len() / 3, "{name}: the enhancement layer is disposable: {disposable:?}");
        }
        let mut dec = filmcraft_vp8::Decoder::new();
        for (i, f) in frames.iter().enumerate() {
            if disposable[i] {
                continue;
            }
            if let Some(p) = dec.decode(f, i as i64).unwrap() {
                let reference = all.iter().find(|(pts, _)| *pts == p.pts).expect("picture of a shown frame");
                assert!(common::i420(&p) == reference.1, "{name}: picture {} changed after leaving out disposable frames", p.pts);
            }
        }
    }
}
