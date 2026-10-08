use std::path::Path;

pub fn normalize_char(c: char) -> Option<&'static str> {
    Some(match c {
        'a'..='z' | 'A'..='Z' | '0'..='9' => return Some(ascii(c)),
        ' ' | '\n' | '.' | ',' | ';' | ':' | '!' | '?' | '-' | '(' | ')' | '"' | '\'' => return Some(ascii(c)),
        'á' => "á", 'à' => "à", 'â' => "â", 'ã' => "ã", 'é' => "é", 'ê' => "ê", 'í' => "í",
        'ó' => "ó", 'ô' => "ô", 'õ' => "õ", 'ú' => "ú", 'ü' => "ü", 'ç' => "ç",
        'Á' => "Á", 'À' => "À", 'Â' => "Â", 'Ã' => "Ã", 'É' => "É", 'Ê' => "Ê", 'Í' => "Í",
        'Ó' => "Ó", 'Ô' => "Ô", 'Õ' => "Õ", 'Ú' => "Ú", 'Ü' => "Ü", 'Ç' => "Ç",
        'è' => "é", 'ò' => "ó", 'ù' => "ú", 'ì' => "í", 'È' => "É",
        'î' | 'ï' => "i", 'û' => "u", 'ñ' => "n", 'æ' => "ae", 'œ' => "oe",
        '«' | '»' | '“' | '”' => "\"", '‘' | '’' => "'", '—' | '–' => "-",
        _ => return None,
    })
}

/// Every character `normalize` can emit.
pub const CHARSET: &str = "\n !\"'(),-.0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzáàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ";

fn ascii(c: char) -> &'static str {
    const TABLE: &str = CHARSET;
    let i = TABLE.find(c).unwrap();
    &TABLE[i..i + 1]
}

pub fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if let Some(r) = normalize_char(c) {
            out.push_str(r);
        }
    }
    out
}

fn body(s: &str) -> &str {
    let start = s.find("*** START OF").and_then(|a| s[a..].find('\n').map(|n| a + n + 1)).unwrap_or(0);
    let end = s.find("*** END OF").unwrap_or(s.len());
    &s[start..end]
}

pub struct Book {
    pub name: String,
    pub train: String,
    pub test: String,
}

/// Every book split 90/10; the test part directly continues the train part.
pub fn books(dir: &Path) -> Vec<Book> {
    let mut files: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
    files.sort();
    files
        .into_iter()
        .map(|f| {
            let raw = std::fs::read_to_string(&f).unwrap().replace('\r', "");
            let chars: Vec<char> = normalize(body(&raw)).chars().collect();
            let cut = chars.len() * 9 / 10;
            let name = f.file_stem().unwrap().to_string_lossy().into_owned();
            Book { name, train: chars[..cut].iter().collect(), test: chars[cut..].iter().collect() }
        })
        .collect()
}

/// Last 10% of every book goes to test, so both splits cover every author.
pub fn load(dir: &Path) -> (String, String) {
    let all = books(dir);
    (all.iter().map(|b| b.train.as_str()).collect(), all.iter().map(|b| b.test.as_str()).collect())
}

/// Extra training-only text: every file in `dir`, Gutenberg header stripped and normalized, in id order.
pub fn extra(dir: &Path) -> String {
    let mut files: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|e| e == "txt")).collect();
    files.sort_by_key(|p| p.file_stem().and_then(|s| s.to_str()).and_then(|s| s.parse::<u64>().ok()).unwrap_or(u64::MAX));
    let mut out = String::new();
    for f in files {
        let raw = std::fs::read_to_string(&f).unwrap_or_default().replace('\r', "");
        out.push_str(&normalize(body(&raw)));
        out.push('\n');
    }
    out
}
