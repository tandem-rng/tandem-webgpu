// The CPU stream kernel: eight chunk lanes stepped row by row, with the lane state in locals so
// the JIT keeps it in registers. Words are int32 inside the loops and wrap on the typed-array
// store.

import { AUX_STREAM, CLOCK_WEYL, DOMAIN_STREAM, type Key, RC } from "./core.ts";

// The views below are little-endian. Every engine that runs this package is.
if (new Uint8Array(new Uint32Array([1]).buffer)[0] !== 1) {
  throw new Error("tandem-webgpu needs a little-endian platform");
}

const MASK32 = 0xffffffffn;

/** High word of a 32 x 32 bit product, from 16-bit halves. Measured against a double product
 * with a rounded quotient, this is 1.8 times faster in the lane loop. */
export function mulHi(a: number, b: number): number {
  const al = a & 0xffff, ah = a >>> 16, bl = b & 0xffff, bh = b >>> 16;
  const lh = al * bh, hl = ah * bl;
  const mid = ((al * bl) >>> 16) + (lh & 0xffff) + (hl & 0xffff);
  return (ah * bh + (lh >>> 16) + (hl >>> 16) + (mid >>> 16)) >>> 0;
}

/** The eight lanes of one chunk group: state o0 o1 o2 o3 h0 h1 h2 h3 per lane, as int32. */
export type Lanes = Int32Array;
export const newLanes = (): Lanes => new Int32Array(64);

/** Seed the lanes of chunk group g: chunk 8g + lane through the seeding function F. */
export function seedGroup(key: Key, g: bigint, st: Lanes): void {
  const c0 = g << 3n, cl = Number(c0 & MASK32) | 0, ch = Number((c0 >> 32n) & MASK32) | 0;
  const k0 = key[0] | 0, k1 = key[1] | 0, k2 = key[2] | 0, k3 = key[3] | 0;
  // core.ts imports this module, so its constants are read at call time, not at load.
  const clock = CLOCK_WEYL | 0, rcs = Int32Array.from(RC);
  for (let lane = 0; lane < 8; lane++) {
    let o0 = cl | lane, o1 = ch, o2 = DOMAIN_STREAM | 0, o3 = AUX_STREAM | 0;
    let h0 = k0, h1 = k1, h2 = k2, h3 = k3;
    for (let r = 0; r < 8; r++) {
      const m0 = h0 | 1, m1 = h1 | 1;
      const lo0 = Math.imul(o0, m0), hi0 = mulHi(o0, m0) | 0;
      const lo1 = Math.imul(o2, m1), hi1 = mulHi(o2, m1) | 0;
      const n0 = o1 ^ hi1 ^ lo1, n1 = ((lo1 << 16) | (lo1 >>> 16)) ^ h2;
      const n2 = o3 ^ hi0 ^ lo0, n3 = ((lo0 << 16) | (lo0 >>> 16)) ^ h3;
      h0 ^= (h1 << 7) | (h1 >>> 25);
      h1 ^= (h2 << 13) | (h2 >>> 19);
      h2 ^= (h3 << 22) | (h3 >>> 10);
      h3 ^= (h0 << 3) | (h0 >>> 29);
      h0 = ((h0 + clock) | 0) ^ n0;
      // The round constant goes on the new o0, then the halves swap.
      const x0 = n0 ^ rcs[r];
      o0 = h0;
      o1 = h1;
      o2 = h2;
      o3 = h3;
      h0 = x0;
      h1 = n1;
      h2 = n2;
      h3 = n3;
    }
    const s = lane * 8;
    st[s] = o0;
    st[s + 1] = o1;
    st[s + 2] = o2;
    st[s + 3] = o3;
    st[s + 4] = h0;
    st[s + 5] = h1;
    st[s + 6] = h2;
    st[s + 7] = h3;
  }
}

/** Step every lane `skip` times without output, then `rows` times writing the exposed half
 * as row r, lane l at dst[off + 32 r + 4 l ..]. The lanes keep their state for the next call. */
export function runRows(
  st: Lanes,
  skip: number,
  rows: number,
  dst: Uint32Array,
  off: number,
): void {
  const total = skip + rows, clock = CLOCK_WEYL | 0;
  for (let lane = 0; lane < 8; lane++) {
    const s = lane * 8;
    let o0 = st[s], o1 = st[s + 1], o2 = st[s + 2], o3 = st[s + 3];
    let h0 = st[s + 4], h1 = st[s + 5], h2 = st[s + 6], h3 = st[s + 7];
    let d = off + lane * 4 - 32 * skip;
    for (let r = 0; r < total; r++) {
      const m0 = h0 | 1, m1 = h1 | 1;
      const lo0 = Math.imul(o0, m0), hi0 = mulHi(o0, m0) | 0;
      const lo1 = Math.imul(o2, m1), hi1 = mulHi(o2, m1) | 0;
      const n0 = o1 ^ hi1 ^ lo1, n1 = ((lo1 << 16) | (lo1 >>> 16)) ^ h2;
      const n2 = o3 ^ hi0 ^ lo0, n3 = ((lo0 << 16) | (lo0 >>> 16)) ^ h3;
      h0 ^= (h1 << 7) | (h1 >>> 25);
      h1 ^= (h2 << 13) | (h2 >>> 19);
      h2 ^= (h3 << 22) | (h3 >>> 10);
      h3 ^= (h0 << 3) | (h0 >>> 29);
      h0 = ((h0 + clock) | 0) ^ n0;
      o0 = n0;
      o1 = n1;
      o2 = n2;
      o3 = n3;
      if (r >= skip) {
        dst[d] = o0;
        dst[d + 1] = o1;
        dst[d + 2] = o2;
        dst[d + 3] = o3;
      }
      d += 32;
    }
    st[s] = o0;
    st[s + 1] = o1;
    st[s + 2] = o2;
    st[s + 3] = o3;
    st[s + 4] = h0;
    st[s + 5] = h1;
    st[s + 6] = h2;
    st[s + 7] = h3;
  }
}

const scratch = newLanes();
const edge = new Uint32Array(32);

/**
 * `n` stream words from word index `first` (bit position 32 first), into out[off ..]. Whole
 * rows are written in place and the cut ends go through one scratch row.
 */
export function streamWords(
  key: Key,
  K: number,
  first: bigint,
  out: Uint32Array,
  off: number,
  n: number,
): void {
  const Kb = BigInt(K);
  let row = first >> 5n, done = 0;
  const skipWords = Number(first & 31n);
  // The row the lanes step to next, so a run that continues a group needs no reseed.
  let lanesAt = -1n;
  const rowsAt = (rows: number, dst: Uint32Array, at: number) => {
    while (rows > 0) {
      const j = Number(row % Kb), take = Math.min(rows, K - j);
      if (lanesAt === row && j > 0) {
        runRows(scratch, 0, take, dst, at);
      } else {
        seedGroup(key, row / Kb, scratch);
        runRows(scratch, j, take, dst, at);
      }
      row += BigInt(take);
      lanesAt = row;
      rows -= take;
      at += 32 * take;
    }
  };
  if (skipWords !== 0) {
    rowsAt(1, edge, 0);
    const m = Math.min(32 - skipWords, n);
    out.set(edge.subarray(skipWords, skipWords + m), off);
    done = m;
  }
  const whole = Math.floor((n - done) / 32);
  if (whole > 0) {
    rowsAt(whole, out, off + done);
    done += 32 * whole;
  }
  if (done < n) {
    rowsAt(1, edge, 0);
    out.set(edge.subarray(0, n - done), off + done);
  }
}
