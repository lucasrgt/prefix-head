use crate::codec::{
    BRAILLE_VOCAB, CAP_BYTE, CELLS_VOCAB, braille_decode, braille_encode, capbytes_decode, capbytes_encode, cells_decode, cells_encode,
    charset_vocab, codepoints_decode, codepoints_encode,
};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

#[derive(Clone, Copy)]
pub enum Base {
    Bytes,
    Braille,
    Cells,
    Codepoints,
    CapBytes,
}

impl Base {
    fn vocab(self) -> usize {
        match self {
            Base::Bytes => 256,
            Base::Braille => BRAILLE_VOCAB,
            Base::Cells => CELLS_VOCAB,
            Base::Codepoints => charset_vocab(),
            Base::CapBytes => CAP_BYTE as usize + 1,
        }
    }

    fn encode(self, s: &str) -> Vec<u32> {
        match self {
            Base::Bytes => s.bytes().map(u32::from).collect(),
            Base::Braille => braille_encode(s),
            Base::Cells => cells_encode(s),
            Base::Codepoints => codepoints_encode(s),
            Base::CapBytes => capbytes_encode(s),
        }
    }

    fn decode(self, syms: &[u32]) -> String {
        match self {
            Base::Bytes => String::from_utf8_lossy(&syms.iter().map(|&s| s as u8).collect::<Vec<u8>>()).into_owned(),
            Base::Braille => braille_decode(syms),
            Base::Cells => cells_decode(syms),
            Base::Codepoints => codepoints_decode(syms),
            Base::CapBytes => capbytes_decode(syms),
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum Crit {
    Freq,
    WordPiece,
    SigGain,
}

impl Crit {
    pub fn suffix(self) -> &'static str {
        match self {
            Crit::Freq => "",
            Crit::WordPiece => "-wp",
            Crit::SigGain => "-sig",
        }
    }

    /// WordPiece: c_xy / (c_x c_y). SigGain (arXiv 2603.19261, main config): z · c_xy^1.25,
    /// z = (c_xy − c_x c_y / N) / sqrt(c_x c_y / N + ε).
    fn score(self, cxy: f64, cx: f64, cy: f64, n: f64) -> f64 {
        match self {
            Crit::Freq => cxy,
            Crit::WordPiece => cxy / (cx * cy),
            Crit::SigGain => {
                let e = cx * cy / n;
                (cxy - e) / (e + 1e-9).sqrt() * cxy.powf(1.25)
            }
        }
    }
}

const MIN_PAIR: i64 = 5;

pub struct Bpe {
    base: Base,
    crit: Crit,
    across: bool,
    merges: HashMap<(u32, u32), (u32, u32)>,
    pieces: Vec<Vec<u32>>,
}

/// Splits before every space/newline. Braille number mode also resets there, so encoding
/// chunk by chunk equals encoding the whole text.
pub(crate) fn chunks(text: &str) -> impl Iterator<Item = &str> {
    let b = text.as_bytes();
    let mut start = 0;
    let mut i = 1;
    std::iter::from_fn(move || {
        if start >= b.len() {
            return None;
        }
        while i < b.len() && b[i] != b' ' && b[i] != b'\n' {
            i += 1;
        }
        let c = &text[start..i];
        start = i;
        i += 1;
        Some(c)
    })
}

fn pairs(w: &[u32]) -> impl Iterator<Item = (u32, u32)> + '_ {
    w.windows(2).map(|p| (p[0], p[1]))
}

/// Greedy merges over (sequence, count) pairs until `pieces` reaches `vocab`.
fn merge_loop(mut words: Vec<(Vec<u32>, u64)>, pieces: &mut Vec<Vec<u32>>, merges: &mut HashMap<(u32, u32), (u32, u32)>, vocab: usize, crit: Crit) {
    let mut count: HashMap<(u32, u32), i64> = HashMap::new();
    let mut where_: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (i, (w, n)) in words.iter().enumerate() {
        for p in pairs(w) {
            *count.entry(p).or_default() += *n as i64;
            where_.entry(p).or_default().push(i);
        }
    }
    let mut heap: BinaryHeap<(i64, Reverse<(u32, u32)>)> = count.iter().map(|(&p, &n)| (n, Reverse(p))).collect();
    let mut sym = vec![0i64; vocab.max(pieces.len())];
    let mut adj = 0i64;
    for (w, n) in &words {
        for &x in w {
            sym[x as usize] += *n as i64;
        }
        adj += (w.len() as i64 - 1) * *n as i64;
    }

    while pieces.len() < vocab {
        let pair = if crit == Crit::Freq {
            let Some((n, Reverse(pair))) = heap.pop() else { break };
            if count.get(&pair).copied().unwrap_or(0) != n || n <= 0 {
                continue;
            }
            pair
        } else {
            let best = count
                .iter()
                .filter(|&(_, &c)| c >= MIN_PAIR)
                .map(|(&p, &c)| (crit.score(c as f64, sym[p.0 as usize] as f64, sym[p.1 as usize] as f64, adj as f64), Reverse(p)))
                .max_by(|a, b| a.partial_cmp(b).unwrap());
            let Some((_, Reverse(pair))) = best else { break };
            pair
        };
        let id = pieces.len() as u32;
        merges.insert(pair, (merges.len() as u32, id));
        pieces.push([pieces[pair.0 as usize].clone(), pieces[pair.1 as usize].clone()].concat());

        let mut touched = where_.remove(&pair).unwrap_or_default();
        touched.dedup();
        let mut changed: HashMap<(u32, u32), ()> = HashMap::new();
        for wi in touched {
            let (w, n) = &mut words[wi];
            if !pairs(w).any(|p| p == pair) {
                continue;
            }
            let n = *n as i64;
            for p in pairs(w) {
                *count.get_mut(&p).unwrap() -= n;
                changed.insert(p, ());
            }
            let mut merged = Vec::with_capacity(w.len());
            let mut i = 0;
            while i < w.len() {
                if i + 1 < w.len() && (w[i], w[i + 1]) == pair {
                    merged.push(id);
                    i += 2;
                } else {
                    merged.push(w[i]);
                    i += 1;
                }
            }
            let hits = (w.len() - merged.len()) as i64 * n;
            sym[pair.0 as usize] -= hits;
            sym[pair.1 as usize] -= hits;
            sym[id as usize] += hits;
            adj -= hits;
            *w = merged;
            for p in pairs(w) {
                *count.entry(p).or_default() += n;
                where_.entry(p).or_default().push(wi);
                changed.insert(p, ());
            }
        }
        for p in changed.into_keys() {
            let n = count[&p];
            if n > 0 {
                if crit == Crit::Freq {
                    heap.push((n, Reverse(p)));
                }
            } else {
                count.remove(&p);
            }
        }
    }
}

impl Bpe {
    pub fn vocab(&self) -> usize {
        self.pieces.len()
    }

    /// Base-alphabet symbols a token stands for.
    pub fn piece(&self, id: u32) -> &[u32] {
        &self.pieces[id as usize]
    }

    /// The (left, right) pair each token was merged from; `None` for base symbols.
    pub fn recipes(&self) -> Vec<Option<(u32, u32)>> {
        let mut r = vec![None; self.pieces.len()];
        for (&pair, &(_, id)) in &self.merges {
            r[id as usize] = Some(pair);
        }
        r
    }

    pub fn crit(&self) -> Crit {
        self.crit
    }

    pub fn base(&self) -> &'static str {
        if self.across {
            return "sbpe";
        }
        match self.base {
            Base::Bytes => "bpe",
            Base::Braille => "bbpe",
            Base::Cells => "g2bpe",
            Base::Codepoints => "cpbpe",
            Base::CapBytes => "capbpe",
        }
    }

    #[cfg(test)]
    pub fn train(text: &str, base: Base, vocab: usize) -> Self {
        Self::train_with(text, base, vocab, Crit::Freq)
    }

    pub fn train_with(text: &str, base: Base, vocab: usize, crit: Crit) -> Self {
        let mut freq: HashMap<&str, u64> = HashMap::new();
        for c in chunks(text) {
            *freq.entry(c).or_default() += 1;
        }
        let mut words: Vec<(Vec<u32>, u64)> = freq.into_iter().map(|(w, n)| (base.encode(w), n)).collect();
        words.sort();

        let mut pieces: Vec<Vec<u32>> = (0..base.vocab() as u32).map(|b| vec![b]).collect();
        let mut merges = HashMap::new();
        merge_loop(words, &mut pieces, &mut merges, vocab, crit);
        Self { base, crit, across: false, merges, pieces }
    }

    /// SuperBPE-style second stage: keeps every word-level merge, then learns merges across spaces
    /// inside lines of `text` until the vocabulary reaches `vocab`. Encoding then works per line.
    pub fn extend_across(&mut self, text: &str, vocab: usize) {
        let mut freq: HashMap<Vec<u32>, u64> = HashMap::new();
        let mut cache = HashMap::new();
        for line in text.split_inclusive('\n') {
            *freq.entry(self.encode_words_cached(line, &mut cache)).or_default() += 1;
        }
        let mut words: Vec<(Vec<u32>, u64)> = freq.into_iter().filter(|(w, _)| w.len() > 1).collect();
        words.sort();
        merge_loop(words, &mut self.pieces, &mut self.merges, vocab, Crit::Freq);
        self.across = true;
    }


    fn apply_merges(&self, w: &mut Vec<u32>) {
        loop {
            let best = pairs(&w).enumerate().filter_map(|(i, p)| self.merges.get(&p).map(|&(rank, id)| (rank, i, id))).min();
            let Some((_, i, id)) = best else { break };
            w[i] = id;
            w.remove(i + 1);
        }
    }

    fn encode_chunk(&self, c: &str) -> Vec<u32> {
        let mut w = self.base.encode(c);
        self.apply_merges(&mut w);
        w
    }

    fn encode_words(&self, text: &str) -> Vec<u32> {
        self.encode_words_cached(text, &mut HashMap::new())
    }

    fn encode_words_cached<'t>(&self, text: &'t str, cache: &mut HashMap<&'t str, Vec<u32>>) -> Vec<u32> {
        let mut out = Vec::new();
        for c in chunks(text) {
            out.extend_from_slice(cache.entry(c).or_insert_with(|| self.encode_chunk(c)));
        }
        out
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if !self.across {
            return self.encode_words(text);
        }
        let mut cache = HashMap::new();
        let mut out = Vec::new();
        for line in text.split_inclusive('\n') {
            let mut w = self.encode_words_cached(line, &mut cache);
            self.apply_merges(&mut w);
            out.extend(w);
        }
        out
    }

    pub fn decode(&self, syms: &[u32]) -> String {
        let base: Vec<u32> = syms.iter().flat_map(|&s| self.pieces[s as usize].iter().copied()).collect();
        self.base.decode(&base)
    }
}
