use std::collections::HashMap;

#[allow(dead_code)]
pub enum Val {
    U(u64),
    I(i64),
    F(f64),
    B(bool),
    S(String),
    Arr(Vec<Val>),
}

pub struct Tensor {
    pub kind: u32,
    pub dims: Vec<usize>,
    pub off: usize,
}

pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q4_0: u32 = 2;
pub const Q8_0: u32 = 8;

pub struct Gguf {
    file: std::fs::File,
    pub meta: HashMap<String, Val>,
    pub tensors: HashMap<String, Tensor>,
    base: usize,
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
    short: bool,
}

const ZEROS: [u8; 16] = [0; 16];

impl Reader<'_> {
    fn take(&mut self, n: usize) -> &[u8] {
        if self.short || self.p + n > self.b.len() {
            self.short = true;
            return if n <= 16 { &ZEROS[..n] } else { &[] };
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        s
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take(8).try_into().unwrap())
    }
    fn str(&mut self) -> String {
        let n = self.u64() as usize;
        String::from_utf8_lossy(self.take(n)).into_owned()
    }
    fn val(&mut self, ty: u32) -> Val {
        match ty {
            0 => Val::U(self.take(1)[0] as u64),
            1 => Val::I(self.take(1)[0] as i8 as i64),
            2 => Val::U(u16::from_le_bytes(self.take(2).try_into().unwrap()) as u64),
            3 => Val::I(i16::from_le_bytes(self.take(2).try_into().unwrap()) as i64),
            4 => Val::U(self.u32() as u64),
            5 => Val::I(self.u32() as i32 as i64),
            6 => Val::F(f32::from_le_bytes(self.take(4).try_into().unwrap()) as f64),
            7 => Val::B(self.take(1)[0] != 0),
            8 => Val::S(self.str()),
            9 => {
                let t = self.u32();
                let n = self.u64() as usize;
                let mut items = Vec::new();
                for _ in 0..n {
                    if self.short {
                        break;
                    }
                    items.push(self.val(t));
                }
                Val::Arr(items)
            }
            10 => Val::U(self.u64()),
            11 => Val::I(self.u64() as i64),
            12 => Val::F(f64::from_le_bytes(self.take(8).try_into().unwrap())),
            _ if self.short => Val::U(0),
            _ => panic!("gguf value type {ty}"),
        }
    }
}

impl Gguf {
    /// Parses the header only; tensor data is read on demand so the file is never held in memory.
    pub fn open(path: &std::path::Path) -> Self {
        use std::io::Read;
        let mut file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut data = Vec::new();
        let mut chunk = vec![0u8; 1 << 22];
        let (meta, tensors, base) = loop {
            let n = file.read(&mut chunk).unwrap();
            data.extend_from_slice(&chunk[..n]);
            if let Some(h) = Self::header(&data) {
                break h;
            }
            assert!(n > 0, "truncated GGUF header");
        };
        Self { file, meta, tensors, base }
    }

    fn header(data: &[u8]) -> Option<(HashMap<String, Val>, HashMap<String, Tensor>, usize)> {
        let mut r = Reader { b: data, p: 0, short: false };
        if r.take(4) != b"GGUF" && !r.short {
            panic!("not a GGUF file");
        }
        let _version = r.u32();
        let (nt, nkv) = (r.u64(), r.u64());
        let mut meta = HashMap::new();
        for _ in 0..nkv {
            let k = r.str();
            let ty = r.u32();
            let v = r.val(ty);
            if r.short {
                return None;
            }
            meta.insert(k, v);
        }
        let mut tensors = HashMap::new();
        for _ in 0..nt {
            let name = r.str();
            let nd = r.u32() as usize;
            let dims = (0..nd).map(|_| r.u64() as usize).collect();
            let kind = r.u32();
            let off = r.u64() as usize;
            if r.short {
                return None;
            }
            tensors.insert(name, Tensor { kind, dims, off });
        }
        let align = match meta.get("general.alignment") {
            Some(Val::U(a)) => *a as usize,
            _ => 32,
        };
        Some((meta, tensors, r.p.div_ceil(align) * align))
    }

    pub fn tensor(&self, name: &str) -> (&Tensor, Vec<u8>) {
        use std::os::unix::fs::FileExt;
        let t = self.tensors.get(name).unwrap_or_else(|| panic!("missing tensor {name}"));
        let n: usize = t.dims.iter().product();
        let bytes = match t.kind {
            F32 => n * 4,
            F16 => n * 2,
            Q8_0 => n / 32 * 34,
            Q4_0 => n / 32 * 18,
            k => panic!("tensor type {k}"),
        };
        let mut buf = vec![0u8; bytes];
        self.file.read_exact_at(&mut buf, (self.base + t.off) as u64).unwrap();
        (t, buf)
    }

    pub fn f32s(&self, name: &str) -> Vec<f32> {
        let (t, b) = self.tensor(name);
        assert_eq!(t.kind, F32);
        b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn f32s_or_empty(&self, name: &str) -> Vec<f32> {
        if self.has(name) { self.f32s(name) } else { Vec::new() }
    }

    pub fn text(&self, key: &str) -> &str {
        match self.meta.get(key) {
            Some(Val::S(v)) => v,
            _ => panic!("missing string {key}"),
        }
    }

    pub fn flag(&self, key: &str) -> bool {
        matches!(self.meta.get(key), Some(Val::B(true)))
    }

    pub fn u_or(&self, key: &str, default: usize) -> usize {
        if self.meta.contains_key(key) { self.u(key) } else { default }
    }

    pub fn u(&self, key: &str) -> usize {
        match self.meta.get(key) {
            Some(Val::U(v)) => *v as usize,
            Some(Val::I(v)) => *v as usize,
            _ => panic!("missing integer {key}"),
        }
    }

    pub fn f(&self, key: &str) -> f32 {
        match self.meta.get(key) {
            Some(Val::F(v)) => *v as f32,
            _ => panic!("missing float {key}"),
        }
    }
}
