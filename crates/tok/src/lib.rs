use bpe::{Base, Bpe, Crit};
use codec::Alphabet;

pub mod bpe;
pub mod codec;
pub mod group;
pub mod syll;
pub mod words;

/// Builds an alphabet by experiment name (`bytes`, `braille`, `bbpe2048-sig`, …), training BPEs on `train`.
pub fn by_name(name: &str, train: &str) -> Alphabet {
    match name {
        "bits" => Alphabet::Bits,
        "nibbles" => Alphabet::Nibbles,
        "braille" => Alphabet::Braille,
        "bytes" => Alphabet::Bytes,
        "g2" => Alphabet::Cells,
        "codepoints" => Alphabet::Codepoints,
        "morse" => Alphabet::Morse,
        "octal" => Alphabet::Octal,
        "capbytes" => Alphabet::CapBytes,
        s if s.starts_with("words") => Alphabet::Words(words::Words::train(train, s["words".len()..].parse().unwrap())),
        s if s.starts_with("sbpe") => {
            let (total, words) = s["sbpe".len()..].split_once('-').expect("sbpe<total>-<word merges>");
            let mut b = Bpe::train_with(train, Base::Bytes, words.parse().unwrap(), Crit::Freq);
            let cut = train.char_indices().nth(20_000_000).map_or(train.len(), |(i, _)| i);
            b.extend_across(&train[..cut], total.parse().unwrap());
            Alphabet::Bpe(b)
        }
        s if ["bpe", "bbpe", "g2bpe", "cpbpe", "capbpe"].iter().any(|p| s.starts_with(p)) => {
            let (head, crit) = match s.split_once('-') {
                Some((h, "sig")) => (h, Crit::SigGain),
                Some((h, "wp")) => (h, Crit::WordPiece),
                None => (s, Crit::Freq),
                _ => panic!("unknown bpe criterion in {s}"),
            };
            let (base, n) = if let Some(n) = head.strip_prefix("bbpe") {
                (Base::Braille, n)
            } else if let Some(n) = head.strip_prefix("g2bpe") {
                (Base::Cells, n)
            } else if let Some(n) = head.strip_prefix("cpbpe") {
                (Base::Codepoints, n)
            } else if let Some(n) = head.strip_prefix("capbpe") {
                (Base::CapBytes, n)
            } else {
                (Base::Bytes, &head[3..])
            };
            Alphabet::Bpe(Bpe::train_with(train, base, n.parse().unwrap(), crit))
        }
        s => panic!("unknown alphabet {s}"),
    }
}
