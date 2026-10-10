//! Persistent spinning thread pool: a matvec per layer is too short for spawn-per-call. Workers that
//! find no work for a while go to sleep, so an idle process (a chat waiting for input) costs nothing.

use std::cell::UnsafeCell;
use std::hint::spin_loop;
use std::sync::{Arc, Condvar, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const IDLE: Duration = Duration::from_millis(50);

type Job = *const (dyn Fn(usize) + Sync);

struct Shared {
    generation: AtomicUsize,
    done: AtomicUsize,
    stop: AtomicBool,
    asleep: AtomicUsize,
    lock: Mutex<()>,
    wake: Condvar,
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
        let shared = Arc::new(Shared { generation: AtomicUsize::new(0), done: AtomicUsize::new(0), stop: AtomicBool::new(false), asleep: AtomicUsize::new(0), lock: Mutex::new(()), wake: Condvar::new(), job: UnsafeCell::new(None) });
        let handles = (1..n)
            .map(|t| {
                let s = shared.clone();
                std::thread::spawn(move || {
                    let mut seen = 0;
                    loop {
                        let mut spins = 0u32;
                        let mut since = None;
                        while s.generation.load(Ordering::Acquire) == seen {
                            spin_loop();
                            spins += 1;
                            if spins > 1 << 16 {
                                std::thread::yield_now();
                                if since.get_or_insert_with(Instant::now).elapsed() > IDLE {
                                    s.asleep.fetch_add(1, Ordering::SeqCst);
                                    let mut guard = s.lock.lock().unwrap();
                                    while s.generation.load(Ordering::SeqCst) == seen {
                                        guard = s.wake.wait(guard).unwrap();
                                    }
                                    drop(guard);
                                    s.asleep.fetch_sub(1, Ordering::SeqCst);
                                }
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
        self.advance();
        f(0);
        while self.shared.done.load(Ordering::Acquire) < self.n - 1 {
            spin_loop();
        }
    }
}

impl Pool {
    fn advance(&self) {
        self.shared.generation.fetch_add(1, Ordering::SeqCst);
        if self.shared.asleep.load(Ordering::SeqCst) > 0 {
            let _guard = self.shared.lock.lock().unwrap();
            self.shared.wake.notify_all();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.advance();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}
