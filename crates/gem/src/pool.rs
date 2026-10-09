//! Persistent spinning thread pool: a matvec per layer is too short for spawn-per-call.

use std::cell::UnsafeCell;
use std::hint::spin_loop;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::JoinHandle;

type Job = *const (dyn Fn(usize) + Sync);

struct Shared {
    generation: AtomicUsize,
    done: AtomicUsize,
    stop: AtomicBool,
    job: UnsafeCell<Option<Job>>,
}

unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct Pool {
    n: usize,
    shared: Arc<Shared>,
    handles: Vec<JoinHandle<()>>,
}

impl Pool {
    pub fn new(n: usize) -> Self {
        let shared = Arc::new(Shared { generation: AtomicUsize::new(0), done: AtomicUsize::new(0), stop: AtomicBool::new(false), job: UnsafeCell::new(None) });
        let handles = (1..n)
            .map(|t| {
                let s = shared.clone();
                std::thread::spawn(move || {
                    let mut seen = 0;
                    loop {
                        let mut spins = 0u32;
                        while s.generation.load(Ordering::Acquire) == seen {
                            spin_loop();
                            spins += 1;
                            if spins > 1 << 16 {
                                std::thread::yield_now();
                            }
                        }
                        seen = s.generation.load(Ordering::Acquire);
                        if s.stop.load(Ordering::Acquire) {
                            return;
                        }
                        let job = unsafe { (*s.job.get()).unwrap() };
                        unsafe { (*job)(t) };
                        s.done.fetch_add(1, Ordering::AcqRel);
                    }
                })
            })
            .collect();
        Self { n, shared, handles }
    }

    pub fn threads(&self) -> usize {
        self.n
    }

    /// Runs `f(t)` for t in 0..threads, the caller doing t = 0, and returns when all are done.
    pub fn run(&self, f: &(dyn Fn(usize) + Sync)) {
        let job: Job = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(f) };
        unsafe { *self.shared.job.get() = Some(job) };
        self.shared.done.store(0, Ordering::Release);
        self.shared.generation.fetch_add(1, Ordering::AcqRel);
        f(0);
        while self.shared.done.load(Ordering::Acquire) < self.n - 1 {
            spin_loop();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.generation.fetch_add(1, Ordering::AcqRel);
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}
