/// Renumbers a vocabulary into `classes × m` codes for a two-factor head: tokens are sorted by
/// their key and every `m` consecutive tokens form one class, so any key gives the same shape and
/// cost and only changes which tokens share a class. Codes past the last token are never produced.
pub struct Grouping {
    pub m: usize,
    pub classes: usize,
    code: Vec<u32>,
    id: Vec<Option<u32>>,
}

impl Grouping {
    pub fn build<K: Ord>(keys: Vec<K>, m: usize) -> Self {
        let mut order: Vec<usize> = (0..keys.len()).collect();
        order.sort_by(|&a, &b| keys[a].cmp(&keys[b]).then(a.cmp(&b)));
        let mut code = vec![0u32; keys.len()];
        for (rank, &t) in order.iter().enumerate() {
            code[t] = rank as u32;
        }
        let classes = keys.len().div_ceil(m);
        let mut id = vec![None; classes * m];
        for (t, &c) in code.iter().enumerate() {
            id[c as usize] = Some(t as u32);
        }
        Self { m, classes, code, id }
    }

    pub fn encode(&self, ids: &[u32]) -> Vec<u32> {
        ids.iter().map(|&t| self.code[t as usize]).collect()
    }

    pub fn decode(&self, codes: &[u32]) -> Vec<u32> {
        codes.iter().filter_map(|&c| self.id[c as usize]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorted_runs_form_classes_and_roundtrip() {
        let keys: Vec<(u32, u32)> = (0..10).map(|i| (i % 3, 10 - i)).collect();
        let g = Grouping::build(keys, 4);
        let ids: Vec<u32> = (0..10).collect();
        assert_eq!(g.decode(&g.encode(&ids)), ids);
        assert_eq!(g.classes, 3);
        let class_of = |t: u32| g.encode(&[t])[0] as usize / 4;
        assert_eq!([9, 6, 3, 0].map(class_of), [0; 4]);
        assert_eq!([7, 4, 1, 8].map(class_of), [1; 4]);
    }
}
