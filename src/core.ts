// Tandem8x32 building blocks on the CPU: the step, the seeding function, key derivation
// and the float mappings. 32-bit arithmetic via Math.imul and >>> 0, 64-bit via BigInt.

export type Key = readonly [number, number, number, number];
export type State = { o: [number, number, number, number]; h: [number, number, number, number] };

const CLOCK_WEYL = 0x9e3779b9;
const DOMAIN_STREAM = 0x9e3779b9;
const DOMAIN_SPLIT = 0xbb67ae85;
const DOMAIN_FORK = 0xd2511f53;
const DOMAIN_FOLD = 0xcd9e8d57;
const DOMAIN_SEED = 0xa54ff53a;
const AUX_STREAM = 0x94d049bb;
const RC = [
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

/** High word of the 64-bit product, from 16-bit halves. */
function mulHi(a: number, b: number): number {
  const al = a & 0xffff, ah = a >>> 16, bl = b & 0xffff, bh = b >>> 16;
  const lh = al * bh, hl = ah * bl;
  const mid = ((al * bl) >>> 16) + (lh & 0xffff) + (hl & 0xffff);
  return (ah * bh + (lh >>> 16) + (hl >>> 16) + (mid >>> 16)) >>> 0;
}

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

/**
 * A stream generator on the CPU: a key, a bit position and a chunk length. Scalar draws,
 * fills and derived generators follow section 5 and 6 of the specification. Each draw
 * costs one `block` call per 16 bytes, so use `fill` on the GPU for bulk output.
 */
export class Tandem {
  #key: Key;
  #position: bigint;
  #K: number;
  #cacheIndex = -1n;
  #cache: Key = [0, 0, 0, 0];

  constructor(key: Key, { position = 0n, K = DEFAULT_K }: { position?: bigint; K?: number } = {}) {
    checkK(K);
    if (position < 0n || position >= POSITION_LIMIT) throw new RangeError("position out of range");
    this.#key = [key[0], key[1], key[2], key[3]];
    this.#position = position;
    this.#K = K;
  }

  static seed(z: bigint, K: number = DEFAULT_K): Tandem {
    return new Tandem(seed(z), { K });
  }

  get key(): Key {
    return this.#key;
  }
  get position(): bigint {
    return this.#position;
  }
  get chunkLength(): number {
    return this.#K;
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

  #draw(w: number): number | bigint {
    const p = this.#span(this.#position, 1n, w);
    const x = this.#read(p, w);
    this.#position = p + BigInt(w);
    return x;
  }

  #each(n: number, w: number, set: (i: number, raw: number | bigint) => void): void {
    const p = this.#span(this.#position, BigInt(n), w);
    for (let i = 0; i < n; i++) set(i, this.#read(p + BigInt(w * i), w));
    this.#position = p + BigInt(w) * BigInt(n);
  }

  #at(i: bigint | number, w: number): number | bigint {
    const k = BigInt(i);
    return this.#read(this.#span(this.#position, k + 1n, w) + BigInt(w) * k, w);
  }

  nextBool(): boolean {
    return this.#draw(1) === 1;
  }
  nextU8(): number {
    return this.#draw(8) as number;
  }
  nextU16(): number {
    return this.#draw(16) as number;
  }
  nextU32(): number {
    return this.#draw(32) as number;
  }
  nextU64(): bigint {
    return this.#draw(64) as bigint;
  }
  nextF32(): number {
    return toFloat32(this.nextU32());
  }
  nextF64(): number {
    return toFloat64(this.nextU64());
  }

  fillU32(n: number): Uint32Array {
    const out = new Uint32Array(n);
    this.#each(n, 32, (i, x) => out[i] = x as number);
    return out;
  }
  fillU64(n: number): BigUint64Array {
    const out = new BigUint64Array(n);
    this.#each(n, 64, (i, x) => out[i] = x as bigint);
    return out;
  }
  fillF32(n: number): Float32Array {
    const out = new Float32Array(n);
    this.#each(n, 32, (i, x) => out[i] = toFloat32(x as number));
    return out;
  }
  fillF64(n: number): Float64Array {
    const out = new Float64Array(n);
    this.#each(n, 64, (i, x) => out[i] = toFloat64(x as bigint));
    return out;
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
    const f = fork(this.#key, this.#position, n);
    this.#position = f.position;
    return f.children.map((key) => new Tandem(key, { K: this.#K }));
  }
}
