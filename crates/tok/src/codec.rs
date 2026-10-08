use crate::bpe::Bpe;
use ruler::corpus::CHARSET;

pub enum Alphabet {
    Morse,
    Octal,
    Codepoints,
    CapBytes,
    Cells,
    Bits,
    Nibbles,
    Braille,
    Bytes,
    Bpe(Bpe),
}

const SPACE: u32 = 0;
const NEWLINE: u32 = 1;
const LETTERS: u32 = 2;
const ACCENTS: &str = "áàâãéêíóôõúüç";
const ACCENT0: u32 = LETTERS + 26;
const PUNCT: &str = ".,;:!?-()\"'";
const PUNCT0: u32 = ACCENT0 + 13;
const CAP: u32 = PUNCT0 + 11;
const NUM: u32 = CAP + 1;
const LETTER_SIGN: u32 = NUM + 1;
pub const BRAILLE_VOCAB: usize = LETTER_SIGN as usize + 1;

fn braille_lower(c: char) -> u32 {
    match c {
        ' ' => SPACE,
        '\n' => NEWLINE,
        'a'..='z' => LETTERS + (c as u32 - 'a' as u32),
        _ => {
            if let Some(i) = ACCENTS.chars().position(|a| a == c) {
                ACCENT0 + i as u32
            } else if let Some(i) = PUNCT.chars().position(|p| p == c) {
                PUNCT0 + i as u32
            } else {
                panic!("no braille cell for {c:?}")
            }
        }
    }
}

fn braille_char(s: u32) -> char {
    match s {
        SPACE => ' ',
        NEWLINE => '\n',
        s if s < ACCENT0 => (b'a' + (s - LETTERS) as u8) as char,
        s if s < PUNCT0 => ACCENTS.chars().nth((s - ACCENT0) as usize).unwrap(),
        s if s < CAP => PUNCT.chars().nth((s - PUNCT0) as usize).unwrap(),
        _ => panic!("control cell {s} has no char"),
    }
}

/// Grade-1 style: capital sign per letter, number sign starts a digit run written with a–j,
/// letter sign separates a digit run from a following a–j.
pub fn braille_encode(text: &str) -> Vec<u32> {
    let mut out = Vec::with_capacity(text.len() * 11 / 10);
    let mut numeric = false;
    for c in text.chars() {
        if let Some(d) = c.to_digit(10) {
            if !numeric {
                out.push(NUM);
                numeric = true;
            }
            out.push(LETTERS + (d + 9) % 10);
            continue;
        }
        if c.is_uppercase() {
            out.push(CAP);
            out.push(braille_lower(c.to_lowercase().next().unwrap()));
        } else {
            let s = braille_lower(c);
            if numeric && (LETTERS..LETTERS + 10).contains(&s) {
                out.push(LETTER_SIGN);
            }
            out.push(s);
        }
        numeric = false;
    }
    out
}

pub fn braille_decode(syms: &[u32]) -> String {
    let mut out = String::new();
    let mut numeric = false;
    let mut cap = false;
    for &s in syms {
        match s {
            NUM => numeric = true,
            CAP => {
                cap = true;
                numeric = false;
            }
            LETTER_SIGN => numeric = false,
            s if numeric && (LETTERS..LETTERS + 10).contains(&s) => {
                out.push(char::from_digit((s - LETTERS + 1) % 10, 10).unwrap());
            }
            s => {
                numeric = false;
                let c = braille_char(s);
                if cap {
                    out.extend(c.to_uppercase());
                    cap = false;
                } else {
                    out.push(c);
                }
            }
        }
    }
    out
}

/// Grade-2 streams from tools/louis_encode.py: 64 six-dot cells, space, newline and markers 0x01–0x04.
pub const CELLS_VOCAB: usize = 70;

fn cell_id(c: char) -> u32 {
    match c {
        '\u{2800}'..='\u{283f}' => c as u32 - 0x2800,
        ' ' => 64,
        '\n' => 65,
        '\u{1}'..='\u{4}' => 65 + c as u32,
        _ => panic!("{c:?} is not a grade-2 stream symbol"),
    }
}

pub fn cells_encode(s: &str) -> Vec<u32> {
    s.chars().map(cell_id).collect()
}

pub fn cells_decode(syms: &[u32]) -> String {
    syms.iter()
        .map(|&s| match s {
            0..64 => char::from_u32(0x2800 + s).unwrap(),
            64 => ' ',
            65 => '\n',
            _ => char::from_u32(s - 65).unwrap(),
        })
        .collect()
}

pub fn charset_vocab() -> usize {
    CHARSET.chars().count()
}

pub fn codepoints_encode(s: &str) -> Vec<u32> {
    s.chars().map(|c| CHARSET.chars().position(|x| x == c).expect("char outside CHARSET") as u32).collect()
}

pub fn codepoints_decode(syms: &[u32]) -> String {
    let set: Vec<char> = CHARSET.chars().collect();
    syms.iter().map(|&s| set[s as usize]).collect()
}

pub const CAP_BYTE: u32 = 256;

pub fn capbytes_encode(s: &str) -> Vec<u32> {
    let mut out = Vec::with_capacity(s.len());
    let mut buf = [0u8; 4];
    for c in s.chars() {
        let c = if c.is_uppercase() {
            out.push(CAP_BYTE);
            c.to_lowercase().next().unwrap()
        } else {
            c
        };
        out.extend(c.encode_utf8(&mut buf).bytes().map(u32::from));
    }
    out
}

pub fn capbytes_decode(syms: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(syms.len());
    let mut caps = Vec::new();
    for &s in syms {
        if s == CAP_BYTE {
            caps.push(bytes.len());
        } else {
            bytes.push(s as u8);
        }
    }
    let text = String::from_utf8_lossy(&bytes);
    let mut out = String::with_capacity(text.len());
    let mut next = caps.iter().peekable();
    for (i, c) in text.char_indices() {
        if next.peek() == Some(&&i) {
            next.next();
            out.extend(c.to_uppercase());
        } else {
            out.push(c);
        }
    }
    out
}

const MORSE: [(char, &str); 49] = [
    ('a', ".-"), ('b', "-..."), ('c', "-.-."), ('d', "-.."), ('e', "."), ('f', "..-."), ('g', "--."),
    ('h', "...."), ('i', ".."), ('j', ".---"), ('k', "-.-"), ('l', ".-.."), ('m', "--"), ('n', "-."),
    ('o', "---"), ('p', ".--."), ('q', "--.-"), ('r', ".-."), ('s', "..."), ('t', "-"), ('u', "..-"),
    ('v', "...-"), ('w', ".--"), ('x', "-..-"), ('y', "-.--"), ('z', "--.."),
    ('0', "-----"), ('1', ".----"), ('2', "..---"), ('3', "...--"), ('4', "....-"), ('5', "....."),
    ('6', "-...."), ('7', "--..."), ('8', "---.."), ('9', "----."),
    ('.', ".-.-.-"), (',', "--..--"), ('?', "..--.."), ('\'', ".----."), ('!', "-.-.--"), ('(', "-.--."),
    (')', "-.--.-"), (':', "---..."), (';', "-.-.-."), ('"', ".-..-."), ('-', "-....-"), ('/', "-..-."), ('&', ".-..."),
];
const M_DOT: u32 = 0;
const M_DASH: u32 = 1;
const M_END: u32 = 2;
const M_SPACE: u32 = 3;
const M_NEWLINE: u32 = 4;
const M_CAP: u32 = 5;
const M_ACCENTS: [(char, char, u32); 13] = [
    ('á', 'a', 6), ('à', 'a', 7), ('â', 'a', 8), ('ã', 'a', 9), ('é', 'e', 6), ('ê', 'e', 8), ('í', 'i', 6),
    ('ó', 'o', 6), ('ô', 'o', 8), ('õ', 'o', 9), ('ú', 'u', 6), ('ü', 'u', 11), ('ç', 'c', 10),
];
const MORSE_VOCAB: usize = 12;

/// ITU Morse with an explicit letter-end symbol; capital and accent marks are separate modifiers.
fn morse_encode(text: &str) -> Vec<u32> {
    let mut out = Vec::with_capacity(text.len() * 4);
    for c in text.chars() {
        match c {
            ' ' => out.push(M_SPACE),
            '\n' => out.push(M_NEWLINE),
            _ => {
                let lower = if c.is_uppercase() {
                    out.push(M_CAP);
                    c.to_lowercase().next().unwrap()
                } else {
                    c
                };
                let (base, accent) = match M_ACCENTS.iter().find(|a| a.0 == lower) {
                    Some(&(_, b, m)) => (b, Some(m)),
                    None => (lower, None),
                };
                let code = MORSE.iter().find(|m| m.0 == base).unwrap_or_else(|| panic!("no morse for {c:?}")).1;
                out.extend(code.bytes().map(|b| if b == b'.' { M_DOT } else { M_DASH }));
                out.extend(accent);
                out.push(M_END);
            }
        }
    }
    out
}

fn morse_decode(syms: &[u32]) -> String {
    let mut out = String::new();
    let (mut code, mut cap, mut accent) = (String::new(), false, None);
    for &s in syms {
        match s {
            M_DOT => code.push('.'),
            M_DASH => code.push('-'),
            M_SPACE => out.push(' '),
            M_NEWLINE => out.push('\n'),
            M_CAP => cap = true,
            M_END => {
                let base = MORSE.iter().find(|m| m.1 == code).map_or('?', |m| m.0);
                let c = accent.and_then(|m| M_ACCENTS.iter().find(|a| a.1 == base && a.2 == m)).map_or(base, |a| a.0);
                if cap {
                    out.extend(c.to_uppercase());
                } else {
                    out.push(c);
                }
                (code, cap, accent) = (String::new(), false, None);
            }
            m => accent = Some(m),
        }
    }
    out
}

impl Alphabet {
    pub fn name(&self) -> String {
        match self {
            Alphabet::Morse => "morse".into(),
            Alphabet::Octal => "octal".into(),
            Alphabet::Codepoints => "codepoints".into(),
            Alphabet::CapBytes => "capbytes".into(),
            Alphabet::Cells => "g2".into(),
            Alphabet::Bits => "bits".into(),
            Alphabet::Nibbles => "nibbles".into(),
            Alphabet::Braille => "braille".into(),
            Alphabet::Bytes => "bytes".into(),
            Alphabet::Bpe(b) => format!("{}{}{}", b.base(), b.vocab(), b.crit().suffix()),
        }
    }

    pub fn vocab(&self) -> usize {
        match self {
            Alphabet::Morse => MORSE_VOCAB,
            Alphabet::Octal => 8,
            Alphabet::Codepoints => charset_vocab(),
            Alphabet::CapBytes => CAP_BYTE as usize + 1,
            Alphabet::Cells => CELLS_VOCAB,
            Alphabet::Bits => 2,
            Alphabet::Nibbles => 16,
            Alphabet::Braille => BRAILLE_VOCAB,
            Alphabet::Bytes => 256,
            Alphabet::Bpe(b) => b.vocab(),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            Alphabet::Morse => morse_encode(text),
            Alphabet::Octal => text.bytes().flat_map(|b| [(b >> 6) as u32, (b >> 3 & 7) as u32, (b & 7) as u32]).collect(),
            Alphabet::Codepoints => codepoints_encode(text),
            Alphabet::CapBytes => capbytes_encode(text),
            Alphabet::Cells => cells_encode(text),
            Alphabet::Bits => text.bytes().flat_map(|b| (0..8).rev().map(move |i| (b >> i) as u32 & 1)).collect(),
            Alphabet::Nibbles => text.bytes().flat_map(|b| [(b >> 4) as u32, (b & 15) as u32]).collect(),
            Alphabet::Braille => braille_encode(text),
            Alphabet::Bytes => text.bytes().map(u32::from).collect(),
            Alphabet::Bpe(b) => b.encode(text),
        }
    }

    pub fn decode(&self, syms: &[u32]) -> String {
        let bytes: Vec<u8> = match self {
            Alphabet::Bits => syms.chunks(8).map(|c| c.iter().fold(0u8, |a, &b| a << 1 | b as u8)).collect(),
            Alphabet::Octal => syms.chunks(3).map(|c| (c[0] << 6 | c[1] << 3 | c[2]) as u8).collect(),
            Alphabet::Morse => return morse_decode(syms),
            Alphabet::Nibbles => syms.chunks(2).map(|c| (c[0] << 4 | c[1]) as u8).collect(),
            Alphabet::Braille => return braille_decode(syms),
            Alphabet::Cells => return cells_decode(syms),
            Alphabet::Codepoints => return codepoints_decode(syms),
            Alphabet::CapBytes => return capbytes_decode(syms),
            Alphabet::Bytes => syms.iter().map(|&s| s as u8).collect(),
            Alphabet::Bpe(b) => return b.decode(syms),
        };
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpe::{Base, Crit};

    #[test]
    fn cells_roundtrip() {
        let s = "\u{1}\u{2828}\u{281d} \u{2}\u{3}\u{4}\n\u{283f}";
        assert_eq!(cells_decode(&cells_encode(s)), s);
        let b = Bpe::train_with(&s.repeat(30), Base::Cells, 80, Crit::Freq);
        assert_eq!(b.decode(&b.encode(s)), s);
    }

    #[test]
    fn every_alphabet_roundtrips() {
        let text = "Capitulo 12 -- \"Dom Casmurro\", 1899b ÁÉ çã!\nNo dia 3a ele disse: (ôô) 'sim'? x0j";
        let bpe = Bpe::train(&text.repeat(20), Base::Bytes, 300);
        let bbpe = Bpe::train(&text.repeat(20), Base::Braille, 100);
        let sig = Bpe::train_with(&text.repeat(20), Base::Braille, 100, Crit::SigGain);
        let wp = Bpe::train_with(&text.repeat(20), Base::Bytes, 300, Crit::WordPiece);
        let cp = Bpe::train_with(&text.repeat(20), Base::Codepoints, 150, Crit::Freq);
        let cap = Bpe::train_with(&text.repeat(20), Base::CapBytes, 320, Crit::Freq);
        let mut sup = Bpe::train_with(&text.repeat(20), Base::Bytes, 290, Crit::Freq);
        sup.extend_across(&text.repeat(20), 330);
        for a in [
            Alphabet::Morse,
            Alphabet::Octal,
            Alphabet::Codepoints,
            Alphabet::CapBytes,
            Alphabet::Bpe(cp),
            Alphabet::Bpe(cap),
            Alphabet::Bpe(sup),
            Alphabet::Bits,
            Alphabet::Nibbles,
            Alphabet::Braille,
            Alphabet::Bytes,
            Alphabet::Bpe(bpe),
            Alphabet::Bpe(bbpe),
            Alphabet::Bpe(sig),
            Alphabet::Bpe(wp),
        ] {
            let syms = a.encode(text);
            assert!(syms.iter().all(|&s| (s as usize) < a.vocab()), "{}", a.name());
            assert_eq!(a.decode(&syms), text, "{}", a.name());
        }
    }
}
