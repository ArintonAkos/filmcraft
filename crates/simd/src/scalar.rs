//! The scalar kernels: the reference every vector version must match, and the code wasm and
//! other targets run.

use crate::{CDEF_COST_WEIGHTS, cdef_pick};

/// Each line sum takes whole rows added at an offset; integer sums don't depend on the order.
pub(crate) fn cdef_direction(rows: &[&[u16; 8]; 8], shift: u32) -> (usize, i32) {
    fn add<const N: usize>(dst: &mut [i32], src: &[i32; N]) {
        for (d, s) in dst.iter_mut().zip(src) {
            *d += s;
        }
    }
    let mut partial = [[0i32; 16]; 8];
    for (i, row) in rows.iter().enumerate() {
        let x: [i32; 8] = std::array::from_fn(|j| (row[j] >> shift) as i32 - 128);
        // Pairs of neighbouring samples, for the directions that step two columns per row.
        let pairs: [i32; 4] = std::array::from_fn(|k| x[2 * k] + x[2 * k + 1]);
        let rev: [i32; 8] = std::array::from_fn(|m| x[7 - m]);
        let rev_pairs: [i32; 4] = std::array::from_fn(|m| pairs[3 - m]);
        add(&mut partial[0][i..], &x);
        add(&mut partial[1][i..], &pairs);
        partial[2][i] = x.iter().sum();
        add(&mut partial[3][i..], &rev_pairs);
        add(&mut partial[4][i..], &rev);
        add(&mut partial[5][3 - i / 2..], &x);
        add(&mut partial[6][..], &x);
        add(&mut partial[7][i / 2..], &x);
    }
    cdef_pick(&std::array::from_fn(|d| partial[d].iter().zip(&CDEF_COST_WEIGHTS[d]).map(|(&p, &w)| p * p * w).sum()))
}
