use bpe::{Base, Bpe, Crit};
use codec::Alphabet;

pub mod bpe;
pub mod codec;
pub mod group;
pub mod syll;

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
