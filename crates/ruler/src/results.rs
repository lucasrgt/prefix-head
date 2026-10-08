use std::io::Write;
use std::path::{Path, PathBuf};

pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn append(file: &str, line: &str) {
    let dir = root().join("results");
    std::fs::create_dir_all(&dir).unwrap();
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(dir.join(file)).unwrap();
    writeln!(f, "{line}").unwrap();
}
