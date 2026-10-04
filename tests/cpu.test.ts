// The CPU path: scalar draws, fills, normals and exponentials against the reference ports.
import { createHash } from "node:crypto";
import { align, block, fillCpu, seed, Tandem } from "../src/mod.ts";
import { COS_POLY, fma32, fma64, horner64, LOG_POLY, SIN_POLY } from "../src/derived.ts";
import { assertEquals, assertThrows, test } from "./harness.ts";
import cross from "./cross.json" with { type: "json" };

const FIXTURE_KEY = cross.CROSS_FILL_KEY as [number, number, number, number];
const bytes = (a: ArrayBufferView) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);

type CExp<T> = [string, T[], string]; // start, 64 values, end position
type DeviceExp = [string, number, number[]]; // start, n, 64 values

test("exponentials equal tandem-c bit for bit, scalar draws and fills", () => {
  for (const [start, want, end] of cross.CROSS_EXPONENTIAL as unknown as CExp<number>[]) {
    const fill = new Tandem(seed(42n), { position: BigInt(start) });
    assertEquals(fill.fillExponentialF64(want.length), Float64Array.from(want), `f64 ${start}`);
    assertEquals(fill.position, BigInt(end));
    const one = new Tandem(seed(42n), { position: BigInt(start) });
    assertEquals(want.map(() => one.nextExponentialF64()), want);
    assertEquals(one.position, BigInt(end));
  }
  for (const [start, want, end] of cross.CROSS_EXPONENTIALF as unknown as CExp<number>[]) {
    const fill = new Tandem(seed(42n), { position: BigInt(start) });
    assertEquals(fill.fillExponentialF32(want.length), Float32Array.from(want), `f32 ${start}`);
    assertEquals(fill.position, BigInt(end));
    const one = new Tandem(seed(42n), { position: BigInt(start) });
    assertEquals(want.map(() => one.nextExponentialF32()), want.map(Math.fround));
  }
});

test("exponentials equal tandem-cuda fills bit for bit", () => {
  for (const [pos, n, want] of cross.CROSS_EXP64 as unknown as DeviceExp[]) {
    const got = new Tandem(FIXTURE_KEY, { position: BigInt(pos) }).fillExponentialF64(n);
    assertEquals(got, Float64Array.from(want.slice(0, n)), `f64 from ${pos}`);
  }
  for (const [pos, n, want] of cross.CROSS_EXP32 as unknown as DeviceExp[]) {
    const got = new Tandem(FIXTURE_KEY, { position: BigInt(pos) }).fillExponentialF32(n);
    assertEquals(got, Float32Array.from(want.slice(0, n)), `f32 from ${pos}`);
  }
});

// The dumps of tandem-c's tools/dump_normals.c and tools/dump_exponentials.c: 1e6 values from
// each of five start positions in both precisions, seed 2026 + 7 2^64, so the hash covers the
// whole polynomial range on both precisions.
const HASH_KEY = seed(2026n + (7n << 64n));
const HASH_STARTS = [0n, 1n, 77n, 12345n, 1n << 30n];
const N = 1_000_000;

test("1e6 normal pairs at five starts hash to the tandem-c dump", () => {
  const hash = createHash("sha256");
  for (const position of HASH_STARTS) {
    const g = new Tandem(HASH_KEY, { position });
    hash.update(bytes(g.fillNormalF64(2 * N - 1)));
    hash.update(bytes(g.fillNormalF32(2 * N - 1)));
  }
  assertEquals(
    hash.digest("hex"),
    "cfae418807a7d5f91ecd3e42c33a00943690c6e4b888ee39206738783efe9ded",
  );
});

test("1e6 exponentials at five starts hash to the tandem-c dump", () => {
  const hash = createHash("sha256");
  for (const position of HASH_STARTS) {
    const g = new Tandem(HASH_KEY, { position });
    hash.update(bytes(g.fillExponentialF64(N)));
    hash.update(bytes(g.fillExponentialF32(N)));
  }
  assertEquals(
    hash.digest("hex"),
    "5c035a4ef1368231d25a9c2f9201be2df3224e28a14549a50625d0db3770ef4e",
  );
});

// A correctly rounded a * b + c from exact integers, the oracle for the emulated fma.
function decompose(x: number): [bigint, number] {
  if (x === 0) return [0n, 0];
  const f = new Float64Array([x]), u = new BigUint64Array(f.buffer);
  const exp = Number((u[0] >> 52n) & 0x7ffn), frac = u[0] & ((1n << 52n) - 1n);
  const m = exp === 0 ? frac : frac | (1n << 52n);
  return [x < 0 ? -m : m, (exp === 0 ? 1 : exp) - 1075];
}

function exactFma(a: number, b: number, c: number, bits: number): number {
  const [ma, ea] = decompose(a), [mb, eb] = decompose(b), [mc, ec] = decompose(c);
  const ep = ea + eb, e = Math.min(ep, ec);
  let n = (ma * mb << BigInt(ep - e)) + (mc << BigInt(ec - e));
  if (n === 0n) return 0;
  const neg = n < 0n;
  if (neg) n = -n;
  const shift = n.toString(2).length - bits;
  let m = n, scale = e;
  if (shift > 0) {
    const s = BigInt(shift), rem = n & ((1n << s) - 1n), half = 1n << (s - 1n);
    m = n >> s;
    if (rem > half || (rem === half && (m & 1n) === 1n)) m += 1n;
    scale += shift;
  }
  return (neg ? -1 : 1) * Number(m) * 2 ** scale;
}

test("the emulated fused multiply-adds are correctly rounded on hard cases", () => {
  // A deterministic generator, so a failure repeats.
  let state = 0x2545f491n;
  const rand = () => {
    state = (state * 6364136223846793005n + 1442695040888963407n) & ((1n << 64n) - 1n);
    return Number(state >> 11n) * 2 ** -53;
  };
  const scaled = (bits: number) => {
    // Random mantissa with some trailing zeros, over a span that keeps Float32 results normal.
    const keep = 1 + Math.floor(rand() * bits), m = Math.floor(rand() * 2 ** keep);
    return (rand() < 0.5 ? -1 : 1) * (m + 1) * 2 ** (Math.floor(rand() * 40) - 20);
  };
  for (const [bits, fma, r] of [[53, fma64, (x: number) => x], [24, fma32, Math.fround]] as const) {
    for (let i = 0; i < 40000; i++) {
      const a = r(scaled(bits)), b = r(scaled(bits)), p = a * b;
      // An unrelated addend, one that cancels the rounded product, and one that leaves a
      // remainder at a random depth below it, so the inner sum is inexact.
      const k = Math.floor(rand() * 50);
      const c = i % 3 === 0
        ? r(scaled(bits))
        : i % 3 === 1
        ? r(-r(p))
        : r(-r(p) + (rand() - 0.5) * Math.abs(p) * 2 ** -(bits + k));
      assertEquals(fma(a, b, c), exactFma(a, b, c, bits), `${bits} bits: ${a} ${b} ${c}`);
    }
  }
});

test("the Float32 fma settles an exact halfway sum by the sign of the lost part", () => {
  // The product of 1 + m 2^-12 with itself has 25 bits, the last one a midpoint of Float32
  // values, and a far smaller addend is lost in the double sum.
  for (const m of [1, 3, 5, 7]) {
    const a = 1 + m * 2 ** -12;
    for (const c of [2 ** -60, -(2 ** -60), 2 ** -70, -(2 ** -70)]) {
      assertEquals(fma32(a, a, c), exactFma(a, a, c, 24), `${a} ${c}`);
    }
  }
});

test("the emulated fma settles an exact halfway sum by the sign of the lost part", () => {
  // a * b = 1 + 2^-26 + 2^-54 and c + 2^-54 is 2^-53 plus 2^-106 below the last place of the
  // sum, so the inner sum rounds and the outer sum sits exactly on a midpoint.
  const a = 1 + 2 ** -27;
  for (const sign of [1, -1]) {
    for (const tail of [1, -1]) {
      const c = sign * (2 ** -54 + tail * 2 ** -106);
      for (const [x, y] of [[a, a], [-a, a], [a, -a]]) {
        assertEquals(fma64(x, y, c), exactFma(x, y, c, 53), `${x} ${y} ${c}`);
      }
    }
  }
});

test("the Horner steps of the polynomials equal the chain of exact fused steps", () => {
  let state = 0x9e3779b9n;
  const rand = () => {
    state = (state * 6364136223846793005n + 1442695040888963407n) & ((1n << 64n) - 1n);
    return Number(state >> 11n) * 2 ** -53;
  };
  // The ranges of the arguments: s^2 of the logarithm, and the squared angle of cos and sin.
  for (const [poly, top] of [[LOG_POLY, 0.0295], [SIN_POLY, 0.62], [COS_POLY, 0.62]] as const) {
    for (let i = 0; i < 20000; i++) {
      const w = i % 50 === 0 ? rand() * 2 ** -40 : rand() * top;
      let acc = poly[0];
      for (let k = 1; k < poly.length; k++) acc = exactFma(w, acc, poly[k], 53);
      assertEquals(horner64(w, poly), acc, `w = ${w}`);
    }
  }
});

test("scalar draws equal the fills across rows, windows and chunk groups", () => {
  for (const K of [1, 8, 32]) {
    const key = seed(3n), n = 3000;
    const u32 = new Tandem(key, { K }).fillU32(n);
    const f64 = new Tandem(key, { K }).fillF64(n / 2);
    const a = new Tandem(key, { K }), b = new Tandem(key, { K }), c = new Tandem(key, { K });
    assertEquals(Uint32Array.from({ length: n }, () => a.nextU32()), u32, `u32 K=${K}`);
    assertEquals(Float64Array.from({ length: n / 2 }, () => b.nextF64()), f64, `f64 K=${K}`);
    assertEquals(
      Array.from({ length: 40 }, () => c.nextU64Pair()),
      Array.from({ length: 40 }, (_, i) => [u32[2 * i], u32[2 * i + 1]]),
    );
    assertEquals(a.position, BigInt(32 * n));
  }
});

test("scalar draws of mixed widths from every bit offset equal the fills", () => {
  const key = seed(9n), count = 300;
  for (const start of [0n, 1n, 7n, 8n, 9n, 31n, 33n, 1000n, 1023n, 1024n, 2049n]) {
    const at = () => new Tandem(key, { position: start });
    const u8 = at().fillU8(count), u16 = at().fillU16(count), bits = at().fillBool(count);
    const a = at(), b = at(), c = at();
    assertEquals(Uint8Array.from({ length: count }, () => a.nextU8()), u8, `u8 from ${start}`);
    assertEquals(Uint16Array.from({ length: count }, () => b.nextU16()), u16, `u16 from ${start}`);
    assertEquals(
      Uint8Array.from({ length: count }, () => c.nextBool() ? 1 : 0),
      bits,
      `bool from ${start}`,
    );
    assertEquals(a.position, align(start, 8) + BigInt(8 * count));
    assertEquals(b.position, align(start, 16) + BigInt(16 * count));
    assertEquals(c.position, start + BigInt(count));
  }
});

test("fills into a caller array equal the allocating fills and touch nothing else", () => {
  const key = seed(5n), start = 77n;
  const fresh = () => new Tandem(key, { position: start });
  const kinds = [
    [Uint32Array, (g: Tandem, o: Uint32Array) => g.fillU32(o), (g: Tandem) => g.fillU32(100)],
    [BigUint64Array, (g: Tandem, o: BigUint64Array) => g.fillU64(o), (g: Tandem) => g.fillU64(100)],
    [Float32Array, (g: Tandem, o: Float32Array) => g.fillF32(o), (g: Tandem) => g.fillF32(100)],
    [Float64Array, (g: Tandem, o: Float64Array) => g.fillF64(o), (g: Tandem) => g.fillF64(100)],
    [
      Float64Array,
      (g: Tandem, o: Float64Array) => g.fillNormalF64(o),
      (g: Tandem) => g.fillNormalF64(100),
    ],
    [
      Float32Array,
      (g: Tandem, o: Float32Array) => g.fillExponentialF32(o),
      (g: Tandem) => g.fillExponentialF32(100),
    ],
    [
      Uint32Array,
      (g: Tandem, o: Uint32Array) => g.fillU32Below(o, 1000),
      (g: Tandem) => g.fillU32Below(100, 1000),
    ],
  ] as const;
  for (const [Ctor, into, make] of kinds) {
    // A view with a nonzero offset in a larger buffer, as a caller's subarray would be.
    const big = new (Ctor as new (n: number) => Uint32Array)(130);
    const view = big.subarray(10, 110) as never;
    const g = fresh();
    (into as (g: Tandem, o: never) => unknown)(g, view);
    const want = make(fresh() as never) as unknown as Uint32Array;
    assertEquals(big.subarray(10, 110), want, Ctor.name);
    assertEquals(big.subarray(0, 10).every((x) => Number(x) === 0), true);
    assertEquals(big.subarray(110).every((x) => Number(x) === 0), true);
    const w = fresh();
    make(w as never);
    assertEquals(g.position, w.position);
  }
});

test("empty fills align the plain kinds and leave the derived kinds where they were", () => {
  const key = seed(6n);
  const run = (fill: (g: Tandem) => unknown) => {
    const g = new Tandem(key, { position: 5n });
    fill(g);
    return g.position;
  };
  assertEquals(run((g) => g.fillU8(0)), 8n);
  assertEquals(run((g) => g.fillU16(0)), 16n);
  assertEquals(run((g) => g.fillU32(0)), 32n);
  assertEquals(run((g) => g.fillU64(0)), 64n);
  assertEquals(run((g) => g.fillF32(0)), 32n);
  assertEquals(run((g) => g.fillF64(0)), 64n);
  assertEquals(run((g) => g.fillBool(0)), 5n);
  for (
    const fill of [
      (g: Tandem) => g.fillU32Below(0, 7),
      (g: Tandem) => g.fillU64Below(0, 7n),
      (g: Tandem) => g.fillNormalF32(0),
      (g: Tandem) => g.fillNormalF64(0),
      (g: Tandem) => g.fillExponentialF32(0),
      (g: Tandem) => g.fillExponentialF64(0),
    ]
  ) {
    assertEquals(run(fill), 5n);
  }
  for (const dtype of ["u8", "i16", "u32", "u64", "f32", "f64", "bool"] as const) {
    assertEquals(fillCpu({ key, count: 0, dtype, position: 5n }).values.length, 0);
  }
});

test("the width of a result-typed bounded draw follows the range", () => {
  const key = seed(12n), edge = 1n << 32n;
  // A range up to 2^32 takes 32-bit draws, whatever the type of the result.
  const a = new Tandem(key), b = new Tandem(key);
  assertEquals(a.nextBelow(1000), b.nextU32Below(1000));
  assertEquals(a.nextBelow(edge), BigInt(b.nextU32Below(2 ** 32)));
  assertEquals(a.position, 64n);
  // Above 2^32 it takes 64-bit draws.
  assertEquals(a.nextBelow(edge + 1n), b.nextU64Below(edge + 1n));
  assertEquals(a.position, b.position);
  assertEquals(
    new Tandem(key).fillBelow(50, 1000),
    new Tandem(key).fillU32Below(50, 1000),
  );
  assertEquals(
    new Tandem(key).fillBelow(50, 1000n),
    BigUint64Array.from(new Tandem(key).fillU32Below(50, 1000), BigInt),
  );
  assertEquals(
    new Tandem(key).fillBelow(50, edge + 1n),
    new Tandem(key).fillU64Below(50, edge + 1n),
  );
});

test("normal and exponential fills cut at an element boundary equal the whole fill", () => {
  const key = seed(21n), start = 37n, cuts = [0, 40, 101, 300];
  for (const K of [32, 8]) {
    const at = (bits: number, count: number) =>
      new Tandem(key, { position: align(start, bits) + BigInt(bits * count), K });
    const fresh = () => new Tandem(key, { position: start, K });
    const w = {
      n64: fresh().fillNormalF64(300),
      n32: fresh().fillNormalF32(300),
      e64: fresh().fillExponentialF64(300),
      e32: fresh().fillExponentialF32(300),
    };
    for (let c = 0; c + 1 < cuts.length; c++) {
      const [lo, hi] = [cuts[c], cuts[c + 1]];
      assertEquals(at(64, lo).fillExponentialF64(hi - lo), w.e64.subarray(lo, hi));
      assertEquals(at(32, lo).fillExponentialF32(hi - lo), w.e32.subarray(lo, hi));
      // A pair is two draws, so a normal fill cuts at even elements.
      const even = lo - (lo % 2), n = Math.min(hi - even, 300 - even);
      assertEquals(at(64, even).fillNormalF64(n), w.n64.subarray(even, even + n), `K=${K} ${lo}`);
      assertEquals(at(32, even).fillNormalF32(n), w.n32.subarray(even, even + n), `K=${K} ${lo}`);
    }
  }
});

test("the stream is the block function for K = 1 and K = 65536", () => {
  const key = seed(2n);
  for (const K of [1, 65536]) {
    for (const first of [0n, 33n, 5000n, 32n * 8n * 65536n - 3n]) {
      const got = new Tandem(key, { position: first * 32n, K }).fillU32(70);
      for (let i = 0; i < 70; i++) {
        const w = first + BigInt(i), row = w >> 5n, lane = (w >> 2n) & 7n;
        const want =
          block(key, 8n * (row / BigInt(K)) + lane, Number(row % BigInt(K)))[Number(w & 3n)];
        assertEquals(got[i], want, `K=${K} word ${w}`);
      }
    }
  }
});

test("draws stop at the 2^64 bound and a fill past it writes nothing", () => {
  const key = seed(4n), top = (1n << 64n) - 64n;
  const g = new Tandem(key, { position: top });
  assertEquals(typeof g.nextU32(), "number");
  assertThrows(() => g.nextU32(), RangeError);
  assertEquals(g.position, top + 32n);
  const out = new Uint32Array(4);
  assertThrows(() => new Tandem(key, { position: top }).fillU32(out), RangeError);
  assertEquals(out, new Uint32Array(4));
  // The last draw that fits ends one word short of 2^64, even when it follows a row boundary.
  const h = new Tandem(key, { position: (1n << 64n) - 1024n - 32n });
  for (let i = 0; i < 32; i++) h.nextU32();
  assertEquals(h.position, (1n << 64n) - 32n);
  assertThrows(() => h.nextU32(), RangeError);
});

test("fillCpu returns the typed arrays and end position of fill, signed types included", () => {
  const key = seed(33n), position = 19n;
  const run = (dtype: "u8" | "i8" | "u16" | "i16" | "u32" | "i32" | "u64" | "i64" | "bool") =>
    fillCpu({ key, position, count: 50, dtype });
  assertEquals(run("i8").values, new Int8Array(new Tandem(key, { position }).fillU8(50).buffer));
  assertEquals(run("i16").values, new Int16Array(new Tandem(key, { position }).fillU16(50).buffer));
  assertEquals(run("i32").values, new Int32Array(new Tandem(key, { position }).fillU32(50).buffer));
  assertEquals(
    run("i64").values,
    new BigInt64Array(new Tandem(key, { position }).fillU64(50).buffer),
  );
  assertEquals(run("bool").position, position + 50n);
  assertEquals(run("u64").position, align(position, 64) + 64n * 50n);
  const below = fillCpu({ key, position, count: 50, dtype: "u64", range: 1000 });
  assertEquals(below.values, new Tandem(key, { position }).fillU64Below(50, 1000n));
  const z = fillCpu({ key, position, count: 51, dtype: "f64", normal: true });
  assertEquals(z.values, new Tandem(key, { position }).fillNormalF64(51));
  assertEquals(z.position, align(position, 64) + 64n * 52n);
  const e = fillCpu({ key, position, count: 51, dtype: "f32", exponential: true });
  assertEquals(e.values, new Tandem(key, { position }).fillExponentialF32(51));
});
