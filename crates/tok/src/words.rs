use crate::bpe::chunks;
use std::collections::HashMap;

/// Word-level vocabulary for the alphabetical-order test (cf. Jiang et al. 2017): the 256 bytes,
/// then the most frequent space-led words of the training text; any other word is spelled in bytes.
pub struct Words {
    pieces: Vec<Vec<u8>>,
    id: HashMap<Vec<u8>, u32>,
}

impl Words {
    pub fn train(text: &str, vocab: usize) -> Self {
        let mut freq: HashMap<&str, u64> = HashMap::new();
        for c in chunks(text) {
            *freq.entry(c).or_default() += 1;
        }
        let mut words: Vec<(&str, u64)> = freq.into_iter().filter(|(w, _)| w.len() > 1).collect();
        words.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        let mut pieces: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        let mut id = HashMap::new();
        for (w, _) in words.into_iter().take(vocab.saturating_sub(256)) {
            id.insert(w.as_bytes().to_vec(), pieces.len() as u32);
            pieces.push(w.as_bytes().to_vec());
        }
        Self { pieces, id }
    }

    pub fn vocab(&self) -> usize {
        self.pieces.len()
    }

    pub fn piece(&self, t: u32) -> &[u8] {
        &self.pieces[t as usize]
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for c in chunks(text) {
            match self.id.get(c.as_bytes()) {
                Some(&t) => out.push(t),
                None => out.extend(c.bytes().map(u32::from)),
            }
        }
        out
    }

    pub fn decode(&self, syms: &[u32]) -> String {
        String::from_utf8_lossy(&syms.iter().flat_map(|&s| self.pieces[s as usize].iter().copied()).collect::<Vec<u8>>()).into_owned()
    }
}
