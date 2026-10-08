mod infer;

use kern::Rng;
use ruler::energy::Rapl;
use kern::dense::*;
use kern::ternary::*;
use std::hint::black_box;
use std::time::{Duration, Instant};

struct Stat {
    ns: f64,
    joules_per_call: Option<f64>,
}

fn measure(rapl: &Option<Rapl>, mut f: impl FnMut()) -> Stat {
    for _ in 0..3 {
        f();
    }
    let budget = Duration::from_millis(600);
    let mut samples = Vec::new();
    let e0 = rapl.as_ref().map(Rapl::read_uj);
    let t0 = Instant::now();
    while t0.elapsed() < budget || samples.len() < 5 {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_nanos() as f64);
    }
    let joules_per_call = match (rapl, e0) {
        (Some(r), Some(e0)) => Some(r.joules_since(e0) / samples.len() as f64),
        _ => None,
    };
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Stat { ns: samples[samples.len() / 2], joules_per_call }
}

fn report(name: &str, rows: usize, cols: usize, bytes: usize, s: Stat) {
    let weights = (rows * cols) as f64;
    let gw = weights / s.ns;
    let gbs = bytes as f64 / s.ns;
    let tok100m = gw * 1e9 / 1e8;
    let nj = s.joules_per_call.map(|j| format!("{:>8.3}", j * 1e9 / weights)).unwrap_or_else(|| "     n/a".into());
    let mb = bytes as f64 / (1 << 20) as f64;
    println!(
        "{name:<16} {rows:>6}x{cols:<6} {mb:>8.1} MB {:>11.1} us {gw:>8.2} Gw/s {gbs:>7.1} GB/s {tok100m:>8.1} tok/s@100M {nj} nJ/w",
        s.ns / 1e3
    );
}

fn main() {
    let rapl = Rapl::open();
    if rapl.is_none() {
        eprintln!("RAPL unreadable: sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj");
    }
    let only = std::env::args().nth(1);
    if only.as_deref() == Some("infer") {
        infer::run(&rapl);
        return;
    }
    let want = |k: &str| only.as_deref().is_none_or(|o| k.contains(o));
    let vnni = has_avx512_vnni();
    let vpop = has_avx512_vpopcnt();
    let avx512f = is_x86_feature_detected!("avx512f");

    for (rows, cols) in [(1024, 1024), (2048, 2048), (4096, 4096), (8192, 16384)] {
        let mut rng = Rng::new(rows as u64 * 31 + cols as u64);
        let x: Vec<i8> = (0..cols).map(|_| rng.i8()).collect();
        let a = Act::new(&x);
        let mut y = vec![0i32; rows];

        let w: Vec<i8> = (0..rows * cols).map(|_| rng.ternary()).collect();
        let t = Ternary::pack(&w, rows, cols);
        drop(w);
        let tb = t.bytes();
        if want("tern_avx2") {
            report("tern_avx2", rows, cols, tb, measure(&rapl, || unsafe { ternary_avx2(&t, &a, black_box(&mut y)) }));
        }
        if vnni && want("tern_vnni") {
            report("tern_vnni", rows, cols, tb, measure(&rapl, || unsafe { ternary_vnni(&t, &a, black_box(&mut y)) }));
            report("tern_vnni_r4", rows, cols, tb, measure(&rapl, || unsafe { ternary_vnni_r4(&t, &a, black_box(&mut y)) }));
        }
        if avx512f && want("tern_sad") {
            report("tern_sad", rows, cols, tb, measure(&rapl, || unsafe { ternary_sad(&t, &a, black_box(&mut y)) }));
        }
        if want("sign_popcnt") {
            report("sign_popcnt", rows, cols, tb, measure(&rapl, || unsafe { sign_popcnt(&t, &a, black_box(&mut y)) }));
        }
        if vpop && want("sign_vpopcnt") {
            report("sign_vpopcnt", rows, cols, tb, measure(&rapl, || unsafe { sign_vpopcnt(&t, &a, black_box(&mut y)) }));
        }
        drop(t);

        if vnni && want("i8_vnni") {
            let w: Vec<i8> = (0..rows * cols).map(|_| rng.i8()).collect();
            let d = DenseI8::new(w, rows, cols);
            report("i8_vnni", rows, cols, rows * cols, measure(&rapl, || unsafe { i8_vnni(&d, &a, black_box(&mut y)) }));
        }
        if avx512f && want("f32_avx512") {
            let wf: Vec<f32> = (0..rows * cols).map(|_| rng.f32()).collect();
            let xf: Vec<f32> = (0..cols).map(|_| rng.f32()).collect();
            let mut yf = vec![0f32; rows];
            report(
                "f32_avx512",
                rows,
                cols,
                rows * cols * 4,
                measure(&rapl, || unsafe { f32_avx512(&wf, rows, cols, &xf, black_box(&mut yf)) }),
            );
        }
        println!();
    }
}
