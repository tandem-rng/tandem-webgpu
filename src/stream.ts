// The CPU stream kernel: eight chunk lanes stepped row by row. The WebAssembly SIMD kernel of
// stream.wat runs where the engine has it. The JavaScript kernel below gives the same words
// elsewhere, with the lane state in locals so the JIT keeps it in registers. Words are int32
// inside the loops and wrap on the typed-array store.

import { AUX_STREAM, CLOCK_WEYL, DOMAIN_STREAM, type Key, RC } from "./core.ts";
import { WASM } from "./stream_wasm.ts";

// The views below are little-endian. Every engine that runs this package is.
if (new Uint8Array(new Uint32Array([1]).buffer)[0] !== 1) {
  throw new Error("tandem-webgpu needs a little-endian platform");
}

const MASK32 = 0xffffffffn;

type Kernel = {
  memory: WebAssembly.Memory;
  out: WebAssembly.Global;
  capacity: WebAssembly.Global;
  seed(): void;
  rows(skip: number, n: number): void;
};

/** The WebAssembly kernel, or undefined where the engine lacks WebAssembly SIMD or a content
 * security policy forbids compiling it. */
function compile(): Kernel | undefined {
  try {
    if (typeof WebAssembly !== "object" || !WebAssembly.validate(WASM)) return undefined;
    return new WebAssembly.Instance(new WebAssembly.Module(WASM)).exports as unknown as Kernel;
  } catch {
    return undefined;
  }
}
const compiled = compile();
let wasm = compiled;
// The memory never grows, so the views stay valid.
const mem = compiled && new Int32Array(compiled.memory.buffer);
const memU = compiled && new Uint32Array(compiled.memory.buffer);
const OUT = compiled ? compiled.out.value >> 2 : 0,
  CAPACITY = compiled ? compiled.capacity.value : 0;

/** Choose the WebAssembly kernel (when the engine has it) or the JavaScript one, and say
 * whether the WebAssembly kernel is in use. Both give the same words. */
export function useWasm(on: boolean): boolean {
  wasm = on ? compiled : undefined;
  return wasm !== undefined;
}

/** High word of a 32 x 32 bit product, from 16-bit halves. Measured against a double product
 * with a rounded quotient, this is 1.8 times faster in the lane loop. */
export function mulHi(a: number, b: number): number {
  const al = a & 0xffff, ah = a >>> 16, bl = b & 0xffff, bh = b >>> 16;
  const lh = al * bh, hl = ah * bl;
  const mid = ((al * bl) >>> 16) + (lh & 0xffff) + (hl & 0xffff);
  return (ah * bh + (lh >>> 16) + (hl >>> 16) + (mid >>> 16)) >>> 0;
}

/** The lanes of one chunk group and where they stand, as int32: word k of lane l at 8 k + l,
 * with words o0 o1 o2 o3 h0 h1 h2 h3, then the key at 64, the chunk counter of lane 0 as low
 * and high word at 68 and 69, the row inside the group at 70 and K at 71. This is the memory
 * layout of the WebAssembly kernel. */
export type Lanes = Int32Array;
export const newLanes = (): Lanes => new Int32Array(72);

/** Seed the lanes of chunk group g at row 0: chunk 8g + lane through the seeding function F. */
export function seedGroup(key: Key, g: bigint, K: number, st: Lanes): void {
  const c0 = g << 3n;
  st.set(key, 64);
  st[68] = Number(c0 & MASK32);
  st[69] = Number((c0 >> 32n) & MASK32);
  st[71] = K;
  if (wasm) {
    mem!.set(st.subarray(64), 64);
    wasm.seed();
    st.set(mem!.subarray(0, 72));
  } else {
    seedLanes(st);
  }
}

function seedLanes(st: Lanes): void {
  const cl = st[68], ch = st[69], k0 = st[64], k1 = st[65], k2 = st[66], k3 = st[67];
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
    st[lane] = o0;
    st[8 + lane] = o1;
    st[16 + lane] = o2;
    st[24 + lane] = o3;
    st[32 + lane] = h0;
    st[40 + lane] = h1;
    st[48 + lane] = h2;
    st[56 + lane] = h3;
  }
  st[70] = 0;
}

/** Step the lanes `skip` times without output, then `rows` times writing the exposed half as
 * row r, lane l at dst[off + 32 r + 4 l ..]. A step at row K of a group seeds the next group
 * first. The lanes keep their state for the next call. */
export function runRows(
  st: Lanes,
  skip: number,
  rows: number,
  dst: Uint32Array,
  off: number,
): void {
  if (wasm) {
    mem!.set(st);
    for (;;) {
      const take = Math.min(rows, CAPACITY);
      wasm.rows(skip, take);
      dst.set(memU!.subarray(OUT, OUT + 32 * take), off);
      rows -= take;
      if (rows === 0) break;
      off += 32 * take;
      skip = 0;
    }
    st.set(mem!.subarray(0, 72));
    return;
  }
  const K = st[71];
  while (skip + rows > 0) {
    if (st[70] === K) {
      const c = (st[68] >>> 0) + 8;
      st[68] = c;
      if (c >= 2 ** 32) st[69]++;
      seedLanes(st);
    }
    const j = st[70], s = Math.min(skip, K - j), r = Math.min(rows, K - j - s);
    stepLanes(st, s, r, dst, off);
    st[70] = j + s + r;
    skip -= s;
    rows -= r;
    off += 32 * r;
  }
}

function stepLanes(st: Lanes, skip: number, rows: number, dst: Uint32Array, off: number): void {
  const total = skip + rows, clock = CLOCK_WEYL | 0;
  for (let lane = 0; lane < 8; lane++) {
    let o0 = st[lane], o1 = st[8 + lane], o2 = st[16 + lane], o3 = st[24 + lane];
    let h0 = st[32 + lane], h1 = st[40 + lane], h2 = st[48 + lane], h3 = st[56 + lane];
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
    st[lane] = o0;
    st[8 + lane] = o1;
    st[16 + lane] = o2;
    st[24 + lane] = o3;
    st[32 + lane] = h0;
    st[40 + lane] = h1;
    st[48 + lane] = h2;
    st[56 + lane] = h3;
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
  const row = first >> 5n, Kb = BigInt(K), skipWords = Number(first & 31n);
  let skip = Number(row % Kb), done = 0;
  seedGroup(key, row / Kb, K, scratch);
  const rowsAt = (rows: number, dst: Uint32Array, at: number) => {
    runRows(scratch, skip, rows, dst, at);
    skip = 0;
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
