use std::time::{Duration, Instant};

const RAPL: &str = "/sys/class/powercap/intel-rapl:0";

/// CPU package energy counter (RAPL). Readable by users only after
/// `sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj`.
pub struct Rapl {
    max_uj: u64,
}

fn read_u64(path: &str) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

impl Rapl {
    pub fn open() -> Option<Self> {
        read_u64(&format!("{RAPL}/energy_uj"))?;
        Some(Self { max_uj: read_u64(&format!("{RAPL}/max_energy_range_uj"))? })
    }

    pub fn read_uj(&self) -> u64 {
        read_u64(&format!("{RAPL}/energy_uj")).unwrap_or(0)
    }

    pub fn joules_since(&self, start_uj: u64) -> f64 {
        let end = self.read_uj();
        let d = if end >= start_uj { end - start_uj } else { end + self.max_uj - start_uj };
        d as f64 / 1e6
    }

    /// Package power with this process asleep, so callers can report marginal energy.
    pub fn idle_watts(&self, window: Duration) -> f64 {
        let e0 = self.read_uj();
        let t0 = Instant::now();
        std::thread::sleep(window);
        self.joules_since(e0) / t0.elapsed().as_secs_f64()
    }

    /// Runs `f`, returning (seconds, package joules).
    pub fn measure(&self, f: impl FnOnce()) -> (f64, f64) {
        let e0 = self.read_uj();
        let t0 = Instant::now();
        f();
        (t0.elapsed().as_secs_f64(), self.joules_since(e0))
    }
}
