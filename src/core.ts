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
