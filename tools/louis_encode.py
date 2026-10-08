"""Word-level grade-2 Portuguese braille via liblouis, made lossless with a grade-1 fallback.

Each space/newline-separated word becomes grade-2 cells when liblouis back-translates it exactly,
otherwise G1 + grade-1 cells, otherwise CH + a prefix-free per-character grade-1 code.
Writes data/norm/{train,test}.g2 (cells, ' ' and '\\n' kept as-is).
"""
import collections, os, subprocess, sys

G1 = '\x01'
CH = '\x02'
OWN = {"'": '\x03', '(': '\x04'}
ROOT = os.path.join(os.path.dirname(__file__), '..')
env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(ROOT, '.tools/lib'))
LOU = os.path.join(ROOT, '.tools/bin/lou_translate')


def run(direction, table, lines):
    out = subprocess.run([LOU, direction, f'unicode.dis,{table}'], input='\n'.join(lines) + '\n',
                         capture_output=True, text=True, env=env, check=True).stdout
    return out.split('\n')[:len(lines)]


def table_for(words, table):
    fw = run('--forward', table, words)
    bw = run('--backward', table, fw)
    return {w: f for w, f, b in zip(words, fw, bw) if b == w}


def main():
    texts = {s: open(os.path.join(ROOT, f'data/norm/{s}.txt')).read() for s in ('train', 'test')}
    counts = collections.Counter(w for t in texts.values() for line in t.split('\n') for w in line.split(' ') if w)
    words = list(counts)
    g2 = table_for(words, 'pt-pt-g2.ctb')
    rest = [w for w in words if w not in g2]
    g1 = table_for(rest, 'pt-pt-g1.utb')
    missing = [w for w in rest if w not in g1]
    tok = sum(counts.values())
    print(f'words {len(words)}: g2 {sum(counts[w] for w in g2) / tok:.2%}, '
          f'g1 {sum(counts[w] for w in g1) / tok:.2%}, per-char {sum(counts[w] for w in missing) / tok:.3%}')

    chars = sorted({c for w in missing for c in w} - set(OWN))
    code = table_for(chars, 'pt-pt-g1.utb')
    if len(code) != len(chars):
        sys.exit(f'chars without exact grade-1 cell: {sorted(set(chars) - set(code))}')
    code.update(OWN)
    vals = list(code.values())
    clash = [(a, b) for a in vals for b in vals if a != b and b.startswith(a)]
    if clash or len(set(vals)) != len(vals):
        sys.exit(f'per-char code is not prefix-free: {clash[:10]}')
    enc = {**{w: CH + ''.join(code[c] for c in w) for w in missing}, **{w: G1 + c for w, c in g1.items()}, **g2}
    cells = set()
    for split, t in texts.items():
        out = '\n'.join(' '.join(enc[w] if w else '' for w in line.split(' ')) for line in t.split('\n'))
        cells.update(out)
        open(os.path.join(ROOT, f'data/norm/{split}.g2'), 'w').write(out)
        print(f'{split}: {len(t)} chars -> {len(out)} symbols ({len(out) / len(t):.3f}/char)')
    print('distinct symbols', len(cells))


main()
