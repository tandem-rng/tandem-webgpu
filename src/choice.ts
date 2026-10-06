// Weighted choice of Appendix C of the specification: Walker's alias method with a table built
// in exact integers, so every port returns the same table and the same indices.

import { mulHi } from "./stream.ts";

const F64 = new Float64Array(1);
const W = new Uint32Array(F64.buffer);
const TWO32 = 2 ** 32;

const nbits = (x: bigint) => x.toString(2).length;

/** Finite w >= 0 as s 2^e with the integer significand s. */
function significand(w: number): [bigint, number] {
  F64[0] = w;
  const biased = W[1] >>> 20, frac = BigInt((W[1] & 0xfffff) * TWO32 + W[0]);
  return biased === 0 ? [frac, -1074] : [frac | (1n << 52n), biased - 1075];
}

/** ceil(w 2^t) for finite w >= 0, exact. ldexp alone would round a positive w to 0 where the
 * product is subnormal. */
function ceilScaled(w: number, t: number): bigint {
  if (w === 0) return 0n;
  const [s, e] = significand(w), k = e + t;
  if (k >= 0) return s << BigInt(k);
  const q = s >> BigInt(-k);
  return (q << BigInt(-k)) === s ? q : q + 1n;
}

/**
 * The alias table of a weighted choice: index i in [0, m) with probability proportional to
 * `weights[i]`. The weights must be finite and not negative, at least one must be positive, and
 * 1 <= m < 2^32, else the constructor throws a RangeError. The build uses exact integers and
 * consumes no draws, so the table equals tandem-c's `tandem_choice_build`. `capacity` is the
 * column capacity S, `cut[j]` the part of column j kept by j, `alias[j]` the index that takes
 * the rest.
 */
export class ChoiceTable {
  readonly capacity: bigint;
  readonly cut: BigUint64Array;
  readonly alias: Uint32Array;

  constructor(weights: ArrayLike<number>) {
    const m = weights.length;
    if (m < 1 || m >= TWO32) throw new RangeError("a choice needs between 1 and 2^32 - 1 weights");
    let wmax = 0;
    for (let i = 0; i < m; i++) {
      const w = weights[i];
      if (!(Number.isFinite(w) && w >= 0)) {
        throw new RangeError("choice weights must be finite and not negative");
      }
      if (w > wmax) wmax = w;
    }
    if (wmax === 0) throw new RangeError("a choice needs a positive weight");
    // The first pass at a scale that cannot overflow bounds the total, and the second puts it
    // just below 2^63.
    const [sig, e] = significand(wmax);
    let t = 63 - (32 - Math.clz32(m)) - (e + nbits(sig) - 1), total = 0n;
    for (let i = 0; i < m; i++) total += ceilScaled(weights[i], t);
    t += 63 - nbits(total);
    const cut = new BigUint64Array(m), alias = new Uint32Array(m);
    let big = 0;
    total = 0n;
    for (let i = 0; i < m; i++) {
      const q = ceilScaled(weights[i], t);
      cut[i] = q;
      total += q;
      if (q > cut[big]) big = i;
      alias[i] = i;
    }
    const M = BigInt(m), S = (total + M - 1n) / M;
    cut[big] += S * M - total;
    // Vose's pairing in place: cut holds the masses until a column is paired.
    let l = 0;
    while (cut[l] < S) l++;
    for (let i = 0; i < m; i++) {
      for (let j = i; j <= i && cut[j] < S;) {
        alias[j] = l;
        cut[l] -= S - cut[j];
        j = l;
        if (cut[l] < S) { do l++; while (l < m && cut[l] < S); }
      }
    }
    this.capacity = S;
    this.cut = cut;
    this.alias = alias;
  }
}

/** The 64-bit draws in `words`, low word first, mapped through the table into `out`. */
export function choiceWords(table: ChoiceTable, words: Uint32Array, out: Uint32Array): void {
  const { alias } = table, m = alias.length, cut = new Uint32Array(table.cut.buffer);
  const sl = Number(table.capacity & 0xffffffffn), sh = Number(table.capacity >> 32n);
  for (let i = 0; i < out.length; i++) {
    const rl = words[2 * i], rh = words[2 * i + 1];
    // x = r m in three words: column j above bit 64, the fraction f = (fh, fl) below.
    const fl = Math.imul(rl, m) >>> 0, s = mulHi(rl, m) + (Math.imul(rh, m) >>> 0);
    const fh = s >>> 0, j = mulHi(rh, m) + Math.floor(s / TWO32);
    // v = (f S) >> 64 from 32-bit products. The partial sums stay below 2^34, exact.
    const t1 = mulHi(fl, sl) + (Math.imul(fl, sh) >>> 0) + (Math.imul(fh, sl) >>> 0);
    const t2 = mulHi(fl, sh) + mulHi(fh, sl) + (Math.imul(fh, sh) >>> 0) + Math.floor(t1 / TWO32);
    const vl = t2 >>> 0, vh = mulHi(fh, sh) + Math.floor(t2 / TWO32);
    const ch = cut[2 * j + 1];
    out[i] = vh < ch || (vh === ch && vl < cut[2 * j]) ? j : alias[j];
  }
}
