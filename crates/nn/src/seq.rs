use crate::kosc::{self, KOsc};
use crate::osc::{self, Osc};
use crate::rnn::{self, MinGru, State};
use crate::sosc::{self, SelOsc};
use std::path::Path;

/// What the training, evaluation and moments code needs from a recurrent core.
pub trait Seq: Sized {
    type Buf;
    type Scratch;
    const KIND: &'static str;
    fn vocab(&self) -> usize;
    fn params(&self) -> usize;
    fn macs_per_token(&self) -> usize;
    fn state_bytes(&self) -> usize;
    fn buf(&self, rows: usize) -> Self::Buf;
    fn forward(&self, x: &[u32], tgt: &[u32], b: usize, t: usize, bf: &mut Self::Buf) -> f64;
    fn backward(&mut self, x: &[u32], tgt: &[u32], b: usize, t: usize, scale: f32, bf: &mut Self::Buf);
    fn carry(&self, b: usize, t: usize, bf: &mut Self::Buf);
    fn reset_stream(&self, s: usize, bf: &mut Self::Buf);
    fn step(&mut self, lr: f32, t: i32, clip: f32) -> f32;
    fn save(&mut self, path: &Path, meta: &str) -> std::io::Result<()>;
    fn load(path: &Path) -> std::io::Result<(Self, String)>;
    fn state(&self) -> State;
    fn scratch(&self) -> Self::Scratch;
    fn advance(&self, id: u32, st: &mut State, s: &mut Self::Scratch);
    fn infer_step(&self, id: u32, st: &mut State, s: &mut Self::Scratch, pick: impl FnMut(usize, &[f32], f32) -> u32, out: &mut [u32]) -> f32;
}

macro_rules! delegate {
    ($t:ty, $m:ident, $kind:literal) => {
        impl Seq for $t {
            type Buf = $m::Buf;
            type Scratch = $m::Scratch;
            const KIND: &'static str = $kind;
            fn vocab(&self) -> usize {
                self.vocab
            }
            fn params(&self) -> usize {
                <$t>::params(self)
            }
            fn macs_per_token(&self) -> usize {
                <$t>::macs_per_token(self)
            }
            fn state_bytes(&self) -> usize {
                <$t>::state_bytes(self)
            }
            fn buf(&self, rows: usize) -> Self::Buf {
                <$t>::buf(self, rows)
            }
            fn forward(&self, x: &[u32], tgt: &[u32], b: usize, t: usize, bf: &mut Self::Buf) -> f64 {
                <$t>::forward(self, x, tgt, b, t, bf)
            }
            fn backward(&mut self, x: &[u32], tgt: &[u32], b: usize, t: usize, scale: f32, bf: &mut Self::Buf) {
                <$t>::backward(self, x, tgt, b, t, scale, bf)
            }
            fn carry(&self, b: usize, t: usize, bf: &mut Self::Buf) {
                <$t>::carry(self, b, t, bf)
            }
            fn reset_stream(&self, s: usize, bf: &mut Self::Buf) {
                <$t>::reset_stream(self, s, bf)
            }
            fn step(&mut self, lr: f32, t: i32, clip: f32) -> f32 {
                <$t>::step(self, lr, t, clip)
            }
            fn save(&mut self, path: &Path, meta: &str) -> std::io::Result<()> {
                <$t>::save(self, path, meta)
            }
            fn load(path: &Path) -> std::io::Result<(Self, String)> {
                <$t>::load(path)
            }
            fn state(&self) -> State {
                <$t>::state(self)
            }
            fn scratch(&self) -> Self::Scratch {
                <$t>::scratch(self)
            }
            fn advance(&self, id: u32, st: &mut State, s: &mut Self::Scratch) {
                <$t>::advance(self, id, st, s)
            }
            fn infer_step(&self, id: u32, st: &mut State, s: &mut Self::Scratch, pick: impl FnMut(usize, &[f32], f32) -> u32, out: &mut [u32]) -> f32 {
                <$t>::infer_step(self, id, st, s, pick, out)
            }
        }
    };
}

delegate!(MinGru, rnn, "mingru");
delegate!(Osc, osc, "osc");
delegate!(SelOsc, sosc, "sosc");
delegate!(KOsc, kosc, "kosc");
