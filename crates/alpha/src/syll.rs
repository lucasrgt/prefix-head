use std::collections::HashMap;

/// Hangul-style blocks for Portuguese: (mod, onset, nucleus, coda).
/// mod = space_before * 3 + case (0 lower, 1 Title, 2 UPPER). Non-letters are blocks whose
/// onset is the symbol itself. Syllable parts come from inventories built on the training text;
/// anything outside them is written one letter per block.
pub struct Syllables {
    onsets: Inv,
    nuclei: Inv,
    codas: Inv,
}

#[derive(Default)]
struct Inv {
    id: HashMap<String, u32>,
    list: Vec<String>,
}

impl Inv {
    fn with(items: impl IntoIterator<Item = String>) -> Self {
        let mut inv = Inv::default();
        inv.add(String::new());
        for s in items {
            inv.add(s);
        }
        inv
    }

    fn add(&mut self, s: String) {
        if !self.id.contains_key(&s) {
            self.id.insert(s.clone(), self.list.len() as u32);
            self.list.push(s);
        }
    }

    fn get(&self, s: &str) -> Option<u32> {
        self.id.get(s).copied()
    }
}

const VOWELS: &str = "aeiouáàâãéêíóôõúü";
const CONSONANTS: &str = "bcdfghjklmnpqrstvwxyzç";
const ONSET2: [&str; 19] = ["pr", "br", "tr", "dr", "cr", "gr", "fr", "vr", "pl", "bl", "cl", "gl", "fl", "tl", "ch", "lh", "nh", "qu", "gu"];
const SYMBOLS: &str = "\n !\"'(),-.0123456789:;?";

fn is_vowel(c: char) -> bool {
    VOWELS.contains(c)
}

/// Splits a lowercase letter run into (onset, nucleus, coda) syllables.
fn syllabify(w: &[char]) -> Vec<(String, String, String)> {
    let mut groups: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < w.len() {
        if is_vowel(w[i]) {
            let s = i;
            while i < w.len() && is_vowel(w[i]) {
                i += 1;
            }
            groups.push((s, i));
        } else {
            i += 1;
        }
    }
    if groups.is_empty() {
        return vec![(w.iter().collect(), String::new(), String::new())];
    }
    let mut cuts = vec![0];
    for p in groups.windows(2) {
        let (a, b) = (p[0].1, p[1].0);
        let cluster: String = w[a..b].iter().collect();
        let onset_len = match b - a {
            1 => 1,
            n if ONSET2.contains(&&cluster[cluster.char_indices().nth(n - 2).unwrap().0..]) => 2,
            _ => 1,
        };
        cuts.push(b - onset_len);
    }
    cuts.push(w.len());
    let mut out = Vec::new();
    for (si, win) in cuts.windows(2).enumerate() {
        let (s, e) = (win[0], win[1]);
        let (ns, ne) = groups[si];
        out.push((w[s..ns].iter().collect(), w[ns..ne].iter().collect(), w[ne..e].iter().collect()));
    }
    out
}

fn case_of(orig: &[char]) -> Option<u32> {
    let up = |c: &char| c.is_uppercase();
    if !orig.iter().any(up) {
        Some(0)
    } else if orig.len() > 1 && orig.iter().all(up) {
        Some(2)
    } else if up(&orig[0]) && !orig[1..].iter().any(up) {
        Some(1)
    } else {
        None
    }
}

impl Syllables {
    pub fn radix(&self) -> Vec<usize> {
        vec![6, self.onsets.list.len(), self.nuclei.list.len(), self.codas.list.len()]
    }

    pub fn train(text: &str) -> Self {
        let mut onsets = Inv::with(CONSONANTS.chars().map(String::from));
        for c in SYMBOLS.chars() {
            onsets.add(format!("\u{0}{c}"));
        }
        let mut nuclei = Inv::with(VOWELS.chars().map(String::from));
        let mut codas = Inv::with(CONSONANTS.chars().map(String::from));
        for word in letter_runs(text) {
            let lower: Vec<char> = word.iter().flat_map(|c| c.to_lowercase()).collect();
            for (o, n, c) in syllabify(&lower) {
                onsets.add(o);
                nuclei.add(n);
                codas.add(c);
            }
        }
        Self { onsets, nuclei, codas }
    }

    fn push_letters(&self, word: &[char], space: &mut bool, out: &mut Vec<u32>) {
        for &c in word {
            let l: String = c.to_lowercase().collect();
            let m = std::mem::take(space) as u32 * 3 + c.is_uppercase() as u32;
            let (o, n) = if is_vowel(l.chars().next().unwrap()) {
                (0, self.nuclei.get(&l).unwrap())
            } else {
                (self.onsets.get(&l).unwrap(), 0)
            };
            out.extend([m, o, n, 0]);
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let chars: Vec<char> = text.chars().collect();
        let mut out = Vec::with_capacity(text.len() * 2);
        let mut space = false;
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if c.is_alphabetic() {
                let s = i;
                while i < chars.len() && chars[i].is_alphabetic() {
                    i += 1;
                }
                let word = &chars[s..i];
                let lower: Vec<char> = word.iter().flat_map(|c| c.to_lowercase()).collect();
                let mut pos = 0;
                for (o, n, cd) in syllabify(&lower) {
                    let len = o.chars().count() + n.chars().count() + cd.chars().count();
                    let orig = &word[pos..pos + len];
                    pos += len;
                    let ids = (self.onsets.get(&o), self.nuclei.get(&n), self.codas.get(&cd), case_of(orig));
                    if let (Some(o), Some(n), Some(cd), Some(case)) = ids {
                        out.extend([std::mem::take(&mut space) as u32 * 3 + case, o, n, cd]);
                    } else {
                        self.push_letters(orig, &mut space, &mut out);
                    }
                }
                continue;
            }
            if c == ' ' && i + 1 < chars.len() && chars[i + 1] != ' ' && !space {
                space = true;
            } else {
                let o = self.onsets.get(&format!("\u{0}{c}")).unwrap_or_else(|| panic!("no block for {c:?}"));
                out.extend([std::mem::take(&mut space) as u32 * 3, o, 0, 0]);
            }
            i += 1;
        }
        assert!(!space, "dangling space");
        out
    }

    pub fn decode(&self, syms: &[u32]) -> String {
        let mut out = String::new();
        for b in syms.chunks(4) {
            if b[0] >= 3 {
                out.push(' ');
            }
            let onset = &self.onsets.list[b[1] as usize];
            if let Some(sym) = onset.strip_prefix('\u{0}') {
                out.push_str(sym);
                continue;
            }
            let s = format!("{onset}{}{}", self.nuclei.list[b[2] as usize], self.codas.list[b[3] as usize]);
            match b[0] % 3 {
                0 => out.push_str(&s),
                1 => {
                    let mut it = s.chars();
                    if let Some(c) = it.next() {
                        out.extend(c.to_uppercase());
                    }
                    out.extend(it);
                }
                _ => out.extend(s.chars().flat_map(char::to_uppercase)),
            }
        }
        out
    }
}

fn letter_runs(text: &str) -> impl Iterator<Item = Vec<char>> + '_ {
    text.split(|c: char| !c.is_alphabetic()).filter(|w| !w.is_empty()).map(|w| w.chars().collect())
}

/// Wraps a flat alphabet as F mixed-radix digits per token (most significant first).
pub fn to_digits(ids: &[u32], radix: &[usize]) -> Vec<u32> {
    let mut out = Vec::with_capacity(ids.len() * radix.len());
    for &id in ids {
        let mut rest = id as usize;
        let mut digits = vec![0u32; radix.len()];
        for q in (0..radix.len()).rev() {
            digits[q] = (rest % radix[q]) as u32;
            rest /= radix[q];
        }
        assert_eq!(rest, 0, "id {id} does not fit radix {radix:?}");
        out.extend(digits);
    }
    out
}

pub fn from_digits(digits: &[u32], radix: &[usize]) -> Vec<u32> {
    digits.chunks(radix.len()).map(|d| d.iter().zip(radix).fold(0usize, |a, (&x, &r)| a * r + x as usize) as u32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syllables_roundtrip() {
        let train = "Capitulo XII. Naquelle tempo, contava apenas quinze annos; MARIA e José, Pedro-a  ver\n";
        let s = Syllables::train(train);
        for t in [train, "Ytrbx QUIÇÁ  ônibus  \n 12 (sr. D'Alcantara) iPhone McDonald's ", " a", "a "] {
            assert_eq!(s.decode(&s.encode(t)), t);
        }
        let blocks = s.encode("Capitulo tempo");
        let names: Vec<String> = blocks.chunks(4).map(|b| s.decode(b)).collect();
        assert_eq!(names, ["Ca", "pi", "tu", "lo", " tem", "po"]);
    }

    #[test]
    fn digits_roundtrip() {
        let r = [8, 8, 8, 16];
        let ids: Vec<u32> = (0..8192).collect();
        assert_eq!(from_digits(&to_digits(&ids, &r), &r), ids);
    }
}
