// Tandem8x32 on the CPU: the step, the seeding function, key derivation, the float mappings
// and the `Tandem` generator. Single steps use Math.imul and >>> 0, positions use BigInt, and
// bulk output comes from the lane kernel in stream.ts.

import {
  type Draws64,
  exponential32,
  exponential64,
  normalPairs32,
  zigFast,
  zigSlow,
} from "./derived.ts";
import { type Lanes, mulHi, newLanes, runRows, seedGroup, streamWords } from "./stream.ts";

export type Key = readonly [number, number, number, number];
export type State = { o: [number, number, number, number]; h: [number, number, number, number] };

export const CLOCK_WEYL = 0x9e3779b9;
export const DOMAIN_STREAM = 0x9e3779b9;
const DOMAIN_SPLIT = 0xbb67ae85;
const DOMAIN_FORK = 0xd2511f53;
const DOMAIN_FOLD = 0xcd9e8d57;
const DOMAIN_SEED = 0xa54ff53a;
export const AUX_STREAM = 0x94d049bb;
export const RC = [
  0xd17cc1b7,
  0xa7220a94,
  0xfe13abe8,
  0xfa9a6ee0,
  0xedb14acc,
  0x9e21c820,
  0xff28b1d5,
  0xef5de2b0,
];

export const DEFAULT_K = 32;
const MASK32 = 0xffffffffn;

const rotl = (x: number, r: number) => ((x << r) | (x >>> (32 - r))) >>> 0;

/** The step T: mix, clock, feedback. Mutates and returns `s`. */
export function T(s: State): State {
  const { o, h } = s;
  const m0 = (h[0] | 1) >>> 0, m1 = (h[1] | 1) >>> 0;
  const lo0 = Math.imul(o[0], m0) >>> 0, hi0 = mulHi(o[0], m0);
  const lo1 = Math.imul(o[2], m1) >>> 0, hi1 = mulHi(o[2], m1);
  const n0 = (o[1] ^ hi1 ^ lo1) >>> 0;
  const n1 = (rotl(lo1, 16) ^ h[2]) >>> 0;
  const n2 = (o[3] ^ hi0 ^ lo0) >>> 0;
  const n3 = (rotl(lo0, 16) ^ h[3]) >>> 0;
  h[0] = (h[0] ^ rotl(h[1], 7)) >>> 0;
  h[1] = (h[1] ^ rotl(h[2], 13)) >>> 0;
  h[2] = (h[2] ^ rotl(h[3], 22)) >>> 0;
  h[3] = (h[3] ^ rotl(h[0], 3)) >>> 0;
  h[0] = (((h[0] + CLOCK_WEYL) >>> 0) ^ n0) >>> 0;
  o[0] = n0;
  o[1] = n1;
  o[2] = n2;
  o[3] = n3;
  return s;
}

/** The seeding function F: eight rounds of T, a round constant, a half swap. */
export function F(s: State): State {
  for (const rc of RC) {
    T(s);
    s.o[0] = (s.o[0] ^ rc) >>> 0;
    [s.o, s.h] = [s.h, s.o];
  }
  return s;
}

export function fKeyed(key: Key, counter: bigint, domain: number, aux: number): State {
  return F({
    o: [Number(counter & MASK32), Number((counter >> 32n) & MASK32), domain, aux],
    h: [key[0], key[1], key[2], key[3]],
  });
}

/** Block B(c, j): the exposed half of chunk c after j + 1 steps. */
export function block(key: Key, c: bigint, j: number): Key {
  const s = fKeyed(key, c, DOMAIN_STREAM, AUX_STREAM);
  for (let i = 0; i <= j; i++) T(s);
  return [...s.o];
}

/** The key of an integer seed in [0, 2^128) through the spec's seed whitening. */
export function seed(z: bigint): Key {
  const h: [number, number, number, number] = [0, 1, 2, 3].map((i) =>
    Number((z >> BigInt(32 * i)) & MASK32)
  ) as [number, number, number, number];
  return [...F({ o: [0, 0, DOMAIN_SEED, 0], h }).o];
}

/** Split child `index` by key alone. */
export function split(key: Key, index: bigint): Key {
  const s = fKeyed(key, index >> 1n, DOMAIN_SPLIT, 0);
  return [...((index & 1n) ? s.h : s.o)];
}

/** A key for a named purpose, by key alone. */
export function sub(key: Key, purpose: bigint): Key {
  return [...fKeyed(key, purpose, DOMAIN_FOLD, 0).o];
}

/** Fork n children from the block at bit `position`. Returns the children and the parent's
 * next position, the start of the following block. */
export function fork(key: Key, position: bigint, n: number): { children: Key[]; position: bigint } {
  const b = position >> 7n;
  const children: Key[] = [];
  for (let i = 0; i < n; i++) {
    const s = fKeyed(key, b, DOMAIN_FORK, i >>> 1);
    children.push([...(i & 1 ? s.h : s.o)]);
  }
  return { children, position: (b + 1n) << 7n };
}

/** Align a bit position to a width. */
export const align = (position: bigint, bits: number) =>
  (position + BigInt(bits) - 1n) & ~(BigInt(bits) - 1n);

/** The spec's Float32 mapping of a 32-bit word: (raw >> 8) * 2^-24. */
export const toFloat32 = (raw: number) => Math.fround((raw >>> 8) * 2 ** -24);

/** The spec's Float64 mapping of a 64-bit value: (raw >> 11) * 2^-53. */
export const toFloat64 = (raw: bigint) => Number(raw >> 11n) * 2 ** -53;

/** Throw unless K is a power of two in [1, 65536]. */
export function checkK(K: number): void {
  if (!Number.isInteger(Math.log2(K)) || K < 1 || K > 65536) {
    throw new RangeError("K must be a power of two in [1, 65536]");
  }
}

const POSITION_LIMIT = 1n << 64n;
const LAST_ROW = POSITION_LIMIT - 1024n;
// Purposes reserved for the fallback generators of bounded fills (Appendix A).
const PURPOSE_BELOW = { 32: 0x424c573332n, 64: 0x424c573634n } as const;
// The purpose reserved for the fallback generators of the Float64 normals (Appendix A).
const PURPOSE_NORMAL64 = 0x4e524d3634n;
// A scalar draw reads this many rows ahead, so a cold position costs one window and not a chunk.
const WINDOW_ROWS = 8;
const RANGE32 = 2 ** 32;
const MASK64 = (1n << 64n) - 1n;
const pair = new Float64Array(2);
const pairF = new Float32Array(2);

/** An output array: the caller's, filled in place, or a new one of length n. */
function target<A extends ArrayBufferView & { length: number }>(
  make: new (n: number) => A,
  n: number | A,
): A {
  return typeof n === "number" ? new make(n) : n;
}

/** The words of a typed array's bytes. Its offset and length are word multiples here. */
const wordsOf = (a: ArrayBufferView) => new Uint32Array(a.buffer, a.byteOffset, a.byteLength >> 2);

/** Uniforms from stream words, in place: the (raw >> 8) 2^-24 and (raw >> 11) 2^-53 maps. */
function mapF32(out: Float32Array, u: Uint32Array, n: number): void {
  for (let i = 0; i < n; i++) out[i] = (u[i] >>> 8) * 2 ** -24;
}
export function mapF64(out: Float64Array, u: Uint32Array, n: number): void {
  for (let i = 0; i < n; i++) out[i] = (u[2 * i + 1] * 2097152 + (u[2 * i] >>> 11)) * 2 ** -53;
}

/** The 64-bit draws of a generator from position 0, one block at a time. A missed normal reads
 * a few draws, so it skips the eight-row window of a `Tandem`. */
class BlockDraws implements Draws64 {
  lo = 0;
  hi = 0;
  #key: Key;
  #K: number;
  #d = 0;
  #b: Key = [0, 0, 0, 0];
  constructor(key: Key, K: number) {
    this.#key = key;
    this.#K = K;
  }
  next(): void {
    // Draw d is words 2 (d & 1) and 2 (d & 1) + 1 of block d >> 1, in row d >> 4.
    const d = this.#d++;
    if ((d & 1) === 0) {
      const row = d >> 4, K = this.#K;
      this.#b = block(this.#key, BigInt(8 * Math.floor(row / K) + ((d >> 1) & 7)), row % K);
    }
    this.lo = this.#b[2 * (d & 1)];
    this.hi = this.#b[2 * (d & 1) + 1];
  }
}

function checkRange32(range: number): void {
  if (!Number.isInteger(range) || range < 0 || range > RANGE32) {
    throw new RangeError("range must be an integer in [0, 2^32]");
  }
}
function checkRange64(range: bigint): void {
  if (range < 0n || range > POSITION_LIMIT) throw new RangeError("range must be in [0, 2^64]");
}

/**
 * A stream generator on the CPU: a key, a bit position and a chunk length. Scalar draws,
 * fills and derived generators follow section 5 and 6 of the specification and Appendix A,
 * on every engine with the same values as the GPU path. Fills take a count and return a new
 * typed array, or take a typed array and fill it in place.
 */
export class Tandem {
  #key: Key;
  #K: number;
  // The position is the bit offset `#off` inside the row that starts at bit `#rowBase`.
  #rowBase = 0n;
  #off = 0;
  #last = false;
  #win: Uint32Array | undefined;
  #winWords = 0; // 0 while the window is stale
  #wi = 0; // word index of the current row inside the window
  #lanes: Lanes | undefined;
  #lanesAt = -1n; // the row the lanes step to next
  #cacheIndex = -1n;
  #cache: Key = [0, 0, 0, 0];
  #zigKey: Key | undefined;

  constructor(key: Key, { position = 0n, K = DEFAULT_K }: { position?: bigint; K?: number } = {}) {
    checkK(K);
    if (position < 0n || position >= POSITION_LIMIT) throw new RangeError("position out of range");
    this.#key = [key[0], key[1], key[2], key[3]];
    this.#K = K;
    this.#setPosition(position);
  }

  static seed(z: bigint, K: number = DEFAULT_K): Tandem {
    return new Tandem(seed(z), { K });
  }

  get key(): Key {
    return this.#key;
  }
  get position(): bigint {
    return this.#rowBase + BigInt(this.#off);
  }
  get chunkLength(): number {
    return this.#K;
  }

  #setPosition(p: bigint): void {
    this.#rowBase = p & ~1023n;
    this.#off = Number(p & 1023n);
    this.#last = this.#rowBase === LAST_ROW;
    this.#winWords = 0;
  }

  /** The bit offset of the next w-bit draw inside its row, with that row in the window. */
  #reserve(w: number): number {
    let o = (this.#off + w - 1) & ~(w - 1);
    if (o + w > 1024) {
      if (this.#last) throw new RangeError("position past 2^64 bits");
      this.#rowBase += 1024n;
      this.#last = this.#rowBase === LAST_ROW;
      this.#wi += 32;
      if (this.#wi >= this.#winWords) this.#winWords = 0;
      o = 0;
    }
    if (this.#last && o + w >= 1024) throw new RangeError("position past 2^64 bits");
    if (this.#winWords === 0) this.#load();
    this.#off = o + w;
    return o;
  }

  /** Fill the window from the row at the position. Stepping on from the last window reuses
   * the lane state, and any other jump reseeds the group. */
  #load(): void {
    const K = BigInt(this.#K), row = this.#rowBase >> 10n;
    const lanes = this.#lanes ??= newLanes(), win = this.#win ??= new Uint32Array(WINDOW_ROWS * 32);
    if (this.#lanesAt === row) {
      runRows(lanes, 0, WINDOW_ROWS, win, 0);
    } else {
      seedGroup(this.#key, row / K, this.#K, lanes);
      runRows(lanes, Number(row % K), WINDOW_ROWS, win, 0);
    }
    this.#lanesAt = row + BigInt(WINDOW_ROWS);
    this.#winWords = 32 * WINDOW_ROWS;
    this.#wi = 0;
  }

  /** The stream block with index n: row n >> 3, lane n & 7. */
  #blockAt(n: bigint): Key {
    if (n !== this.#cacheIndex) {
      const row = n >> 3n, K = BigInt(this.#K);
      this.#cache = block(this.#key, 8n * (row / K) + (n & 7n), Number(row % K));
      this.#cacheIndex = n;
    }
    return this.#cache;
  }

  /** The w-bit value at aligned bit position p, as a bigint for w = 64 and a number below. */
  #read(p: bigint, w: number): number | bigint {
    const b = this.#blockAt(p >> 7n), o = Number(p & 127n), i = o >>> 5;
    if (w === 64) return BigInt(b[i]) | (BigInt(b[i + 1]) << 32n);
    return w === 32 ? b[i] : (b[i] >>> (o & 31)) & ((1 << w) - 1);
  }

  /** The aligned start of a run of n draws of width w, checked against the 2^64 bound. */
  #span(from: bigint, n: bigint, w: number): bigint {
    const p = align(from, w);
    if (p + BigInt(w) * n >= POSITION_LIMIT) {
      throw new RangeError("position past 2^64 bits");
    }
    return p;
  }

  #at(i: bigint | number, w: number): number | bigint {
    const k = BigInt(i);
    return this.#read(this.#span(this.position, k + 1n, w) + BigInt(w) * k, w);
  }

  /** One word of the current row's window, after a #reserve. */
  #word(o: number): number {
    return this.#win![this.#wi + (o >> 5)];
  }

  nextBool(): boolean {
    const o = this.#reserve(1);
    return ((this.#word(o) >>> (o & 31)) & 1) === 1;
  }
  nextU8(): number {
    const o = this.#reserve(8);
    return (this.#word(o) >>> (o & 31)) & 0xff;
  }
  nextU16(): number {
    const o = this.#reserve(16);
    return (this.#word(o) >>> (o & 31)) & 0xffff;
  }
  nextU32(): number {
    return this.#word(this.#reserve(32));
  }
  /** A 64-bit draw as its low and high word, which needs no BigInt. */
  nextU64Pair(): [number, number] {
    const o = this.#reserve(64), i = this.#wi + (o >> 5), win = this.#win!;
    return [win[i], win[i + 1]];
  }
  nextU64(): bigint {
    const [lo, hi] = this.nextU64Pair();
    return (BigInt(hi) << 32n) | BigInt(lo);
  }
  nextF32(): number {
    return (this.nextU32() >>> 8) * 2 ** -24;
  }
  nextF64(): number {
    const [lo, hi] = this.nextU64Pair();
    return (hi * 2097152 + (lo >>> 11)) * 2 ** -53;
  }

  /** Stream words for `count` draws of width w into `out`, then move past them. Checks the
   * endpoint before it writes. Returns the aligned start bit. */
  #stream(out: Uint32Array, count: number, w: number): bigint {
    const p = this.#span(this.position, BigInt(count), w);
    streamWords(this.#key, this.#K, p >> 5n, out, 0, out.length);
    this.#setPosition(p + BigInt(w) * BigInt(count));
    return p;
  }

  /** A fill of w-bit values smaller than a word, cut out of the covering words. */
  #fillBytes<A extends ArrayBufferView & { length: number }>(
    make: new (n: number) => A,
    n: number | A,
    w: 8 | 16,
  ): A {
    const out = target(make, n), count = out.length;
    const p = this.#span(this.position, BigInt(count), w);
    const skip = Number(p & 31n) >> 3, bytes = count * (w >> 3);
    const tmp = new Uint32Array((skip + bytes + 3) >> 2);
    streamWords(this.#key, this.#K, p >> 5n, tmp, 0, tmp.length);
    new Uint8Array(out.buffer, out.byteOffset, out.byteLength).set(
      new Uint8Array(tmp.buffer, skip, bytes),
    );
    this.#setPosition(p + BigInt(w) * BigInt(count));
    return out;
  }

  fillU8(n: number | Uint8Array): Uint8Array {
    return this.#fillBytes(Uint8Array, n, 8);
  }
  fillU16(n: number | Uint16Array): Uint16Array {
    return this.#fillBytes(Uint16Array, n, 16);
  }
  fillU32(n: number | Uint32Array): Uint32Array {
    const out = target(Uint32Array, n);
    this.#stream(out, out.length, 32);
    return out;
  }
  fillU64(n: number | BigUint64Array): BigUint64Array {
    const out = target(BigUint64Array, n);
    this.#stream(wordsOf(out), out.length, 64);
    return out;
  }
  /** Bits of the stream from the position, one 0 or 1 per element. */
  fillBool(n: number | Uint8Array): Uint8Array {
    const out = target(Uint8Array, n), count = out.length, p = this.position;
    if (p + BigInt(count) >= POSITION_LIMIT) throw new RangeError("position past 2^64 bits");
    const skip = Number(p & 31n), tmp = new Uint32Array((skip + count + 31) >> 5);
    streamWords(this.#key, this.#K, p >> 5n, tmp, 0, tmp.length);
    for (let i = 0; i < count; i++) out[i] = (tmp[(skip + i) >> 5] >>> ((skip + i) & 31)) & 1;
    this.#setPosition(p + BigInt(count));
    return out;
  }
  fillF32(n: number | Float32Array): Float32Array {
    const out = target(Float32Array, n);
    this.#stream(wordsOf(out), out.length, 32);
    mapF32(out, wordsOf(out), out.length);
    return out;
  }
  fillF64(n: number | Float64Array): Float64Array {
    const out = target(Float64Array, n);
    this.#stream(wordsOf(out), out.length, 64);
    mapF64(out, wordsOf(out), out.length);
    return out;
  }

  /** One draw in [0, range) by Lemire's method, rejecting on the draws that follow. A range of
   * 0 gives 0 and consumes one draw. */
  nextU32Below(range: number): number {
    checkRange32(range);
    const x = this.nextU32();
    if (range === RANGE32) return x;
    let lo = Math.imul(x, range) >>> 0, hi = mulHi(x, range);
    if (lo < range) {
      const t = (RANGE32 - range) % range;
      while (lo < t) {
        const y = this.nextU32();
        lo = Math.imul(y, range) >>> 0;
        hi = mulHi(y, range);
      }
    }
    return hi;
  }
  nextU64Below(range: bigint): bigint {
    checkRange64(range);
    let m = this.nextU64() * range;
    if ((m & MASK64) < range) {
      const t = (POSITION_LIMIT - range) % range;
      while ((m & MASK64) < t) m = this.nextU64() * range;
    }
    return m >> 64n;
  }

  /** A bounded draw whose width follows the range, as the specification asks of an interface
   * that names only the result type: 32-bit draws up to 2^32, else 64-bit. The width does not
   * change the values. */
  nextBelow(range: number): number;
  nextBelow(range: bigint): bigint;
  nextBelow(range: number | bigint): number | bigint {
    if (typeof range === "number") {
      if (range > RANGE32) throw new RangeError("a number range is at most 2^32, use a bigint");
      return this.nextU32Below(range);
    }
    return range <= BigInt(RANGE32)
      ? BigInt(this.nextU32Below(Number(range)))
      : this.nextU64Below(range);
  }

  /**
   * A bounded fill of values in [0, range), as in Appendix A. It consumes exactly one draw per
   * element, and a rejected draw retries on the fallback generator of its global draw index,
   * so a fill cut at any element boundary equals the whole fill. An empty fill moves nothing.
   */
  fillU32Below(n: number | Uint32Array, range: number): Uint32Array {
    checkRange32(range);
    const out = target(Uint32Array, n), count = out.length;
    if (count === 0) return out;
    const g0 = this.#stream(out, count, 32) >> 5n;
    if (range === RANGE32) return out;
    if (range === 0) return out.fill(0);
    const t = (RANGE32 - range) % range;
    for (let i = 0; i < count; i++) {
      const x = out[i], lo = Math.imul(x, range) >>> 0;
      if (lo < t) out[i] = this.#retry32(g0 + BigInt(i), range, t);
      else out[i] = mulHi(x, range);
    }
    return out;
  }
  fillU64Below(n: number | BigUint64Array, range: bigint): BigUint64Array {
    checkRange64(range);
    const out = target(BigUint64Array, n), count = out.length, u = wordsOf(out);
    if (count === 0) return out;
    const g0 = this.#stream(u, count, 64) >> 6n;
    if (range === POSITION_LIMIT) return out;
    if (range === 0n) return out.fill(0n);
    const t = (POSITION_LIMIT - range) % range;
    const rl = Number(range & MASK32), rh = Number(range >> 32n);
    const tl = Number(t & MASK32), th = Number(t >> 32n);
    // The 128-bit product x * range in words w0 to w3, from 32-bit products, since a BigInt
    // product per element is ten times slower. The partial sums stay below 2^34, exact.
    for (let i = 0; i < count; i++) {
      const xl = u[2 * i], xh = u[2 * i + 1], w0 = Math.imul(xl, rl) >>> 0;
      const s1 = mulHi(xl, rl) + (Math.imul(xl, rh) >>> 0) + (Math.imul(xh, rl) >>> 0);
      const w1 = s1 >>> 0;
      if (w1 < th || (w1 === th && w0 < tl)) {
        out[i] = this.#retry64(g0 + BigInt(i), range, t);
        continue;
      }
      const s2 = mulHi(xl, rh) + mulHi(xh, rl) + (Math.imul(xh, rh) >>> 0) +
        Math.floor(s1 / RANGE32);
      u[2 * i] = s2 >>> 0;
      u[2 * i + 1] = mulHi(xh, rh) + Math.floor(s2 / RANGE32);
    }
    return out;
  }
  /** The result-typed bounded fill: a Uint32Array for a number range, a BigUint64Array for a
   * bigint one, with the draw width chosen from the range as in `nextBelow`. */
  fillBelow(n: number | Uint32Array, range: number): Uint32Array;
  fillBelow(n: number | BigUint64Array, range: bigint): BigUint64Array;
  fillBelow(
    n: number | Uint32Array | BigUint64Array,
    range: number | bigint,
  ): Uint32Array | BigUint64Array {
    if (typeof range === "number") return this.fillU32Below(n as number | Uint32Array, range);
    if (range > BigInt(RANGE32)) return this.fillU64Below(n as number | BigUint64Array, range);
    const out = target(BigUint64Array, n as number | BigUint64Array);
    const narrow = this.fillU32Below(out.length, Number(range));
    for (let i = 0; i < out.length; i++) out[i] = BigInt(narrow[i]);
    return out;
  }

  /** The draws of the fallback generator for global draw index g, until one is accepted. */
  #fallback(w: 32 | 64, g: bigint): Tandem {
    return new Tandem(split(sub(this.#key, PURPOSE_BELOW[w]), g), { K: this.#K });
  }
  #retry32(g: bigint, range: number, t: number): number {
    const r = this.#fallback(32, g);
    for (;;) {
      const x = r.nextU32();
      if ((Math.imul(x, range) >>> 0) >= t) return mulHi(x, range);
    }
  }
  #retry64(g: bigint, range: bigint, t: bigint): bigint {
    const r = this.#fallback(64, g);
    for (;;) {
      const m = r.nextU64() * range;
      if ((m & MASK64) >= t) return m >> 64n;
    }
  }

  /**
   * A standard normal by the 1024-layer ziggurat of Appendix A from one 64-bit draw. A draw
   * that misses the inner rectangles continues on the fallback generator of its global draw
   * index, which leaves this generator's position alone. The values equal tandem-c bit for bit.
   */
  nextNormalF64(): number {
    const g = align(this.position, 64) >> 6n, [lo, hi] = this.nextU64Pair();
    const z = zigFast(lo, hi);
    return z === z ? z : zigSlow(lo, hi, this.#zigFallback(g));
  }
  /**
   * A standard normal pair from two Float32 uniform draws by Box-Muller, cosine half first, in
   * single precision with the polynomials of tandem-c, within 4 ulps of its values.
   */
  nextNormal2F32(): [number, number] {
    pairF[0] = this.nextF32();
    pairF[1] = this.nextF32();
    normalPairs32(pairF, 1);
    return [pairF[0], pairF[1]];
  }
  /** The cosine half of the pair, which consumes both draws. */
  nextNormalF32(): number {
    return this.nextNormal2F32()[0];
  }

  /**
   * Standard normals by the ziggurat. Element i comes from 64-bit draw i alone, so a fill cut
   * at any element equals the whole fill. An empty fill aligns the position to 64 bits.
   */
  fillNormalF64(n: number | Float64Array): Float64Array {
    const out = target(Float64Array, n), count = out.length, u = wordsOf(out);
    const g = this.#stream(u, count, 64) >> 6n;
    // Element i overwrites the words of draw i after reading them.
    for (let i = 0; i < count; i++) {
      const lo = u[2 * i], hi = u[2 * i + 1], z = zigFast(lo, hi);
      out[i] = z === z ? z : zigSlow(lo, hi, this.#zigFallback(g + BigInt(i)));
    }
    return out;
  }
  /** The fallback generator of a missed normal at global draw index g. */
  #zigFallback(g: bigint): Draws64 {
    this.#zigKey ??= sub(this.#key, PURPOSE_NORMAL64);
    return new BlockDraws(split(this.#zigKey, g), this.#K);
  }

  /**
   * Standard normals by Box-Muller. Elements 2j and 2j + 1 come from uniform draws 2j and
   * 2j + 1, so an odd n still consumes both draws of its last pair. An empty fill moves
   * nothing.
   */
  fillNormalF32(n: number | Float32Array): Float32Array {
    const out = target(Float32Array, n), count = out.length, pairs = count >> 1;
    if (count === 0) return out;
    this.#uniformsF32(out, pairs * 2);
    normalPairs32(out, pairs);
    if (count & 1) out[count - 1] = this.nextNormal2F32()[0];
    return out;
  }

  /** Standard exponentials -ln(1 - u) of one uniform each, with the polynomial logarithm of
   * tandem-c, within 4 ulps of its values. A fill consumes one draw per element. */
  nextExponentialF64(): number {
    pair[0] = this.nextF64();
    exponential64(pair, 1);
    return pair[0];
  }
  nextExponentialF32(): number {
    pairF[0] = this.nextF32();
    exponential32(pairF, 1);
    return pairF[0];
  }
  fillExponentialF64(n: number | Float64Array): Float64Array {
    const out = target(Float64Array, n);
    if (out.length === 0) return out;
    this.#uniformsF64(out, out.length);
    exponential64(out, out.length);
    return out;
  }
  fillExponentialF32(n: number | Float32Array): Float32Array {
    const out = target(Float32Array, n);
    if (out.length === 0) return out;
    this.#uniformsF32(out, out.length);
    exponential32(out, out.length);
    return out;
  }

  /** The first m elements of `out` as the uniforms of the plain fill. */
  #uniformsF64(out: Float64Array, m: number): void {
    const u = wordsOf(out), p = this.#span(this.position, BigInt(m), 64);
    streamWords(this.#key, this.#K, p >> 5n, u, 0, 2 * m);
    mapF64(out, u, m);
    this.#setPosition(p + 64n * BigInt(m));
  }
  #uniformsF32(out: Float32Array, m: number): void {
    const u = wordsOf(out), p = this.#span(this.position, BigInt(m), 32);
    streamWords(this.#key, this.#K, p >> 5n, u, 0, m);
    mapF32(out, u, m);
    this.#setPosition(p + 32n * BigInt(m));
  }

  /** Element i of the fill that would start here. The position does not move. */
  atU32(i: bigint | number): number {
    return this.#at(i, 32) as number;
  }
  atU64(i: bigint | number): bigint {
    return this.#at(i, 64) as bigint;
  }
  atF32(i: bigint | number): number {
    return toFloat32(this.atU32(i));
  }
  atF64(i: bigint | number): number {
    return toFloat64(this.atU64(i));
  }

  /** Child `index` by key alone, at position 0 with this generator's K. */
  split(index: bigint | number): Tandem {
    return new Tandem(split(this.#key, BigInt(index)), { K: this.#K });
  }

  /** A child for a named purpose, by key alone. */
  sub(purpose: bigint | number): Tandem {
    return new Tandem(sub(this.#key, BigInt(purpose)), { K: this.#K });
  }

  /** Fork n children from the current block and advance this generator to the next one. */
  fork(n: number): Tandem[] {
    const f = fork(this.#key, this.position, n);
    this.#setPosition(f.position);
    return f.children.map((key) => new Tandem(key, { K: this.#K }));
  }
}
