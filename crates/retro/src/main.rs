//! Retrofits a frozen LLM with a two-level output head: P(t | h) = P(class | h) · P(t | class, h).
//! P(t | class, h) is the softmax of the model's own output rows restricted to the class, so it is
//! exact; only the class predictor (classes × d) is trained, on true next tokens of real text.
//! Inputs come from `.tools/hidden` (final-norm hidden states) and `tools/hidden/head.py` (output rows).

use kern::Rng;
use nn::train::{Param, gemm};
use std::path::{Path, PathBuf};

fn f16(x: u16) -> f32 {
    let sign = if x & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((x >> 10) & 0x1f) as i32;
    let mant = (x & 0x3ff) as f32;
    match exp {
        0 => sign * mant * 2f32.powi(-24),
        31 => sign * f32::INFINITY,
        _ => sign * (1.0 + mant / 1024.0) * 2f32.powi(exp - 15),
    }
}

fn read_f16(path: &Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    b.chunks_exact(2).map(|c| f16(u16::from_le_bytes([c[0], c[1]]))).collect()
}

fn field(json: &str, key: &str) -> usize {
    let at = json.find(&format!("\"{key}\":")).unwrap_or_else(|| panic!("missing {key}")) + key.len() + 3;
    json[at..].chars().take_while(char::is_ascii_digit).collect::<String>().parse().unwrap()
}

struct Dump {
    h: Vec<f32>,
    y: Vec<u32>,
    chars: usize,
}

fn load(prefix: &Path) -> (Dump, usize, usize) {
    let json = std::fs::read_to_string(prefix.with_extension("json")).unwrap();
    let (d, v) = (field(&json, "d"), field(&json, "vocab"));
    let y: Vec<u32> = std::fs::read(prefix.with_extension("y")).unwrap().chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let h = read_f16(&prefix.with_extension("h"));
    assert_eq!(h.len(), y.len() * d);
    (Dump { h, y, chars: field(&json, "chars") }, d, v)
}

fn pieces(path: &Path) -> Vec<Vec<u8>> {
    let b = std::fs::read(path).unwrap();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let n = b[i] as usize;
        out.push(b[i + 1..i + 1 + n].to_vec());
        i += 1 + n;
    }
    out
}

fn chunked(order: &[u32], v: usize, m: usize) -> Vec<u32> {
    let mut class = vec![0u32; v];
    for (pos, &t) in order.iter().enumerate() {
        class[t as usize] = (pos / m) as u32;
    }
    class
}

/// Balanced k-means on the output rows: Lloyd iterations, then a greedy capacity-limited assignment
/// over each token's 16 nearest centroids.
fn semantic(w: &[f32], d: usize, v: usize, m: usize, seed: u64) -> Vec<u32> {
    let k = v.div_ceil(m);
    let mut rng = Rng::new(seed);
    let mut keys: Vec<(u64, usize)> = (0..v).map(|t| (rng.next(), t)).collect();
    keys.sort();
    let mut cent: Vec<f32> = keys[..k].iter().flat_map(|&(_, t)| w[t * d..(t + 1) * d].iter().copied()).collect();
    let block = 8192;
    let dists = |cent: &[f32], f: &mut dyn FnMut(usize, &[f32])| {
        let norm: Vec<f32> = cent.chunks(d).map(|c| c.iter().map(|x| x * x).sum()).collect();
        let mut s = vec![0f32; block * k];
        for start in (0..v).step_by(block) {
            let n = block.min(v - start);
            gemm(n, d, k, &w[start * d..(start + n) * d], false, cent, true, 0.0, &mut s[..n * k]);
            for r in 0..n {
                let row = &mut s[r * k..(r + 1) * k];
                for (x, nc) in row.iter_mut().zip(&norm) {
                    *x = nc - 2.0 * *x;
                }
                f(start + r, row);
            }
        }
    };
    let mut assign = vec![0usize; v];
    for _ in 0..12 {
        dists(&cent, &mut |t, row| {
            assign[t] = (0..k).min_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
        });
        let mut sum = vec![0f32; k * d];
        let mut n = vec![0usize; k];
        for t in 0..v {
            n[assign[t]] += 1;
            for (s, x) in sum[assign[t] * d..(assign[t] + 1) * d].iter_mut().zip(&w[t * d..(t + 1) * d]) {
                *s += x;
            }
        }
        for c in 0..k {
            if n[c] == 0 {
                let t = (rng.next() % v as u64) as usize;
                cent[c * d..(c + 1) * d].copy_from_slice(&w[t * d..(t + 1) * d]);
            } else {
                for (x, s) in cent[c * d..(c + 1) * d].iter_mut().zip(&sum[c * d..(c + 1) * d]) {
                    *x = s / n[c] as f32;
                }
            }
        }
    }
    let top = 16.min(k);
    let mut pairs: Vec<(f32, u32, u32)> = Vec::with_capacity(v * top);
    dists(&cent, &mut |t, row| {
        let mut idx: Vec<usize> = (0..k).collect();
        idx.select_nth_unstable_by(top - 1, |&a, &b| row[a].total_cmp(&row[b]));
        pairs.extend(idx[..top].iter().map(|&c| (row[c], t as u32, c as u32)));
    });
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut class, mut fill) = (vec![u32::MAX; v], vec![0usize; k]);
    for (_, t, c) in pairs {
        if class[t as usize] == u32::MAX && fill[c as usize] < m {
            class[t as usize] = c;
            fill[c as usize] += 1;
        }
    }
    let mut free: Vec<u32> = (0..k as u32).flat_map(|c| std::iter::repeat_n(c, m - fill[c as usize])).collect();
    for c in class.iter_mut().filter(|c| **c == u32::MAX) {
        *c = free.pop().unwrap();
    }
    class
}

fn grouping(kind: &str, w: &[f32], d: usize, v: usize, m: usize, pieces: &[Vec<u8>], train_y: &[u32], seed: u64) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..v as u32).collect();
    match kind {
        "prefix" => ids.sort_by(|&a, &b| pieces[a as usize].cmp(&pieces[b as usize]).then(a.cmp(&b))),
        "freq" => {
            let mut n = vec![0u64; v];
            for &y in train_y {
                n[y as usize] += 1;
            }
            ids.sort_by(|&a, &b| n[b as usize].cmp(&n[a as usize]).then(a.cmp(&b)));
        }
        "id" => {}
        "random" => {
            let mut rng = Rng::new(seed);
            let keys: Vec<u64> = (0..v).map(|_| rng.next()).collect();
            ids.sort_by_key(|&t| keys[t as usize]);
        }
        "sem" => return semantic(w, d, v, m, seed),
        _ => panic!("grouping: prefix|freq|id|random|sem"),
    }
    chunked(&ids, v, m)
}

fn lse(row: &[f32]) -> f32 {
    let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    mx + row.iter().map(|x| (x - mx).exp()).sum::<f32>().ln()
}

/// Nats of the full softmax per row, cached next to the dump.
fn full_nats(test: &Dump, w: &[f32], d: usize, v: usize, cache: &Path) -> Vec<f32> {
    if let Ok(b) = std::fs::read(cache) {
        return b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    }
    let rows = test.y.len();
    let block = 64;
    let mut s = vec![0f32; block * v];
    let mut out = Vec::with_capacity(rows);
    for start in (0..rows).step_by(block) {
        let n = block.min(rows - start);
        gemm(n, d, v, &test.h[start * d..(start + n) * d], false, w, true, 0.0, &mut s[..n * v]);
        for r in 0..n {
            let row = &s[r * v..(r + 1) * v];
            out.push(lse(row) - row[test.y[start + r] as usize]);
        }
    }
    std::fs::write(cache, out.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    out
}

/// Nats of P(t | class, h) per row with the original output rows.
fn within_nats(test: &Dump, w: &[f32], d: usize, class: &[u32], members: &[Vec<u32>]) -> Vec<f32> {
    let rows = test.y.len();
    let mut by_class: Vec<Vec<usize>> = vec![Vec::new(); members.len()];
    for i in 0..rows {
        by_class[class[test.y[i] as usize] as usize].push(i);
    }
    let mut out = vec![0f32; rows];
    for (c, idx) in by_class.iter().enumerate() {
        if idx.is_empty() {
            continue;
        }
        let mem = &members[c];
        let wc: Vec<f32> = mem.iter().flat_map(|&t| w[t as usize * d..(t as usize + 1) * d].iter().copied()).collect();
        let hc: Vec<f32> = idx.iter().flat_map(|&i| test.h[i * d..(i + 1) * d].iter().copied()).collect();
        let mut s = vec![0f32; idx.len() * mem.len()];
        gemm(idx.len(), d, mem.len(), &hc, false, &wc, true, 0.0, &mut s);
        for (r, &i) in idx.iter().enumerate() {
            let row = &s[r * mem.len()..(r + 1) * mem.len()];
            let at = mem.iter().position(|&t| t == test.y[i]).unwrap();
            out[i] = lse(row) - row[at];
        }
    }
    out
}

struct Classifier {
    a: Param,
    b: Param,
    k: usize,
    d: usize,
}

impl Classifier {
    /// Returns summed nats; with `train`, accumulates gradients of the mean loss.
    fn step(&mut self, h: &[f32], target: &[u32], train: bool) -> f64 {
        let (n, k, d) = (target.len(), self.k, self.d);
        let mut s = vec![0f32; n * k];
        gemm(n, d, k, h, false, &self.a.w, true, 0.0, &mut s);
        let mut nats = 0f64;
        for r in 0..n {
            let row = &mut s[r * k..(r + 1) * k];
            for (x, b) in row.iter_mut().zip(&self.b.w) {
                *x += b;
            }
            let l = lse(row);
            nats += (l - row[target[r] as usize]) as f64;
            if train {
                for x in row.iter_mut() {
                    *x = (*x - l).exp() / n as f32;
                }
                row[target[r] as usize] -= 1.0 / n as f32;
            }
        }
        if train {
            gemm(k, n, d, &s, true, h, false, 0.0, &mut self.a.g);
            self.b.g.iter_mut().for_each(|g| *g = 0.0);
            for r in 0..n {
                for (g, x) in self.b.g.iter_mut().zip(&s[r * k..(r + 1) * k]) {
                    *g += x;
                }
            }
        }
        nats
    }

    fn eval(&mut self, dump: &Dump, class: &[u32]) -> f64 {
        let d = self.d;
        let mut nats = 0f64;
        for start in (0..dump.y.len()).step_by(1024) {
            let end = (start + 1024).min(dump.y.len());
            let t: Vec<u32> = dump.y[start..end].iter().map(|&y| class[y as usize]).collect();
            nats += self.step(&dump.h[start * d..end * d], &t, false);
        }
        nats
    }
}

fn arg<T: std::str::FromStr>(args: &[String], key: &str, default: T) -> T {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tag: String = arg(&args, "--model", "gemma270m".to_string());
    let kind: String = arg(&args, "--group", "prefix".to_string());
    let epochs: f64 = arg(&args, "--epochs", 3.0);
    let lr: f32 = arg(&args, "--lr", 2e-3);
    let batch: usize = arg(&args, "--batch", 512);
    let seed: u64 = arg(&args, "--seed", 1);
    let dir: PathBuf = ruler::results::root().join("data/retro");

    let (train, d, v) = load(&dir.join(format!("{tag}-train")));
    let (test, _, _) = load(&dir.join(format!("{tag}-test")));
    let m: usize = arg(&args, "--m", (v as f64).sqrt().ceil() as usize);
    let w = read_f16(&dir.join(format!("{tag}.W")));
    assert_eq!(w.len(), v * d);
    let pieces = pieces(&dir.join(format!("{tag}-test.vocab")));

    let class = grouping(&kind, &w, d, v, m, &pieces, &train.y, seed);
    let k = *class.iter().max().unwrap() as usize + 1;
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); k];
    for t in 0..v {
        members[class[t] as usize].push(t as u32);
    }
    eprintln!("{tag} {kind}: vocab {v}, d {d}, {k} classes of <= {m}, train {} rows, test {} rows", train.y.len(), test.y.len());

    let full = full_nats(&test, &w, d, v, &dir.join(format!("{tag}-test.full")));
    let within = within_nats(&test, &w, d, &class, &members);

    let mut init = vec![0f32; k * d];
    for (c, mem) in members.iter().enumerate() {
        for &t in mem {
            for (x, y) in init[c * d..(c + 1) * d].iter_mut().zip(&w[t as usize * d..(t as usize + 1) * d]) {
                *x += y / mem.len() as f32;
            }
        }
    }
    let mut rng = Rng::new(seed);
    let mut a = Param::new(k * d, 0.0, &mut rng);
    a.w = init;
    let mut clf = Classifier { a, b: Param::new(k, 0.0, &mut rng), k, d };
    let centroid = clf.eval(&test, &class);

    let rows = train.y.len();
    let steps = (epochs * rows as f64 / batch as f64) as usize;
    let mut order: Vec<usize> = (0..rows).collect();
    let mut hb = vec![0f32; batch * d];
    let mut yb = vec![0u32; batch];
    let t0 = std::time::Instant::now();
    let mut run = 0f64;
    for step in 0..steps {
        let at = (step * batch) % rows;
        if at < batch {
            let keys: Vec<u64> = (0..rows).map(|_| rng.next()).collect();
            order.sort_by_key(|&i| keys[i]);
        }
        for j in 0..batch {
            let i = order[(at + j) % rows];
            hb[j * d..(j + 1) * d].copy_from_slice(&train.h[i * d..(i + 1) * d]);
            yb[j] = class[train.y[i] as usize];
        }
        run += clf.step(&hb, &yb, true) / batch as f64;
        let rate = lr * (1.0 - step as f32 / steps as f32);
        clf.a.adam(rate, step as i32 + 1);
        clf.b.adam(rate, step as i32 + 1);
        if (step + 1) % (steps / 10).max(1) == 0 {
            eprintln!("  step {}/{steps} class loss {:.4} nats/token  {:.0}s", step + 1, run / (steps / 10).max(1) as f64, t0.elapsed().as_secs_f64());
            run = 0.0;
        }
    }
    let trained = clf.eval(&test, &class);

    let bits = |x: f64| x / std::f64::consts::LN_2 / test.chars as f64;
    let sum = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>();
    let (full_b, within_b, class_b, centroid_b) = (bits(sum(&full)), bits(sum(&within)), bits(trained), bits(centroid));
    let oracle_b = full_b - within_b;
    let total = class_b + within_b;
    let line = format!(
        "{{\"model\":\"{tag}\",\"group\":\"{kind}\",\"vocab\":{v},\"d\":{d},\"classes\":{k},\"m\":{m},\"seed\":{seed},\"epochs\":{epochs},\"train_rows\":{rows},\"test_rows\":{},\"test_chars\":{},\"full_bpc\":{full_b:.4},\"factored_bpc\":{total:.4},\"class_bpc\":{class_b:.4},\"class_oracle_bpc\":{oracle_b:.4},\"within_bpc\":{within_b:.4},\"centroid_class_bpc\":{centroid_b:.4},\"head_macs_full\":{},\"head_macs_factored\":{}}}",
        test.y.len(),
        test.chars,
        v * d,
        (k + m) * d
    );
    println!("{line}");
    ruler::results::append("retro.jsonl", &line);
}
