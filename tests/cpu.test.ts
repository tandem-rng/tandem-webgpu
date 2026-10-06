// The CPU path: scalar draws, fills, normals and exponentials against the reference ports.
import { createHash } from "node:crypto";
import { align, block, fillCpu, seed, Tandem } from "../src/mod.ts";
import { fma64, horner64, LOG_POLY, neg2Log64 } from "../src/derived.ts";
import { useWasm } from "../src/stream.ts";
import { ZIG_K } from "../src/zig_tables.ts";
import { assertEquals, assertThrows, test } from "./harness.ts";
import cross from "./cross.json" with { type: "json" };

const FIXTURE_KEY = cross.CROSS_FILL_KEY as [number, number, number, number];
const bytes = (a: ArrayBufferView) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);

type CExp<T> = [string, T[], string]; // start, 64 values, end position
type DeviceExp = [string, number, number[]]; // start, n, 64 values

// The exponentials and the Float32 normals round each multiply-add twice where tandem-c fuses
// it. The bounds are the ulp tolerance of Appendix A.
const near64 = (got: number, want: number) => Math.abs(got - want) <= 4 * 2 ** -52 * Math.abs(want);
const near32 = (got: number, want: number) =>
  Math.abs(got - Math.fround(want)) <= 4 * 2 ** -23 * Math.abs(want) + 1e-6;
const allNear = (near: (a: number, b: number) => boolean, got: ArrayLike<number>, want: number[]) =>
  want.every((w, i) => near(got[i], w));

test("exponentials agree with tandem-c within 4 ulps, scalar draws equal the fills", () => {
  for (const [start, want, end] of cross.CROSS_EXPONENTIAL as unknown as CExp<number>[]) {
    const fill = new Tandem(seed(42n), { position: BigInt(start) });
    const got = fill.fillExponentialF64(want.length);
    assertEquals(allNear(near64, got, want), true, `f64 ${start}`);
    assertEquals(fill.position, BigInt(end));
    const one = new Tandem(seed(42n), { position: BigInt(start) });
    assertEquals(Float64Array.from(want, () => one.nextExponentialF64()), got);
    assertEquals(one.position, BigInt(end));
  }
  for (const [start, want, end] of cross.CROSS_EXPONENTIALF as unknown as CExp<number>[]) {
    const fill = new Tandem(seed(42n), { position: BigInt(start) });
    const got = fill.fillExponentialF32(want.length);
    assertEquals(allNear(near32, got, want), true, `f32 ${start}`);
    assertEquals(fill.position, BigInt(end));
    const one = new Tandem(seed(42n), { position: BigInt(start) });
    assertEquals(Float32Array.from(want, () => one.nextExponentialF32()), got);
  }
});

test("exponentials agree with tandem-cuda fills within 4 ulps", () => {
  for (const [pos, n, want] of cross.CROSS_EXP64 as unknown as DeviceExp[]) {
    const got = new Tandem(FIXTURE_KEY, { position: BigInt(pos) }).fillExponentialF64(n);
    assertEquals(allNear(near64, got, want.slice(0, n)), true, `f64 from ${pos}`);
  }
  for (const [pos, n, want] of cross.CROSS_EXP32 as unknown as DeviceExp[]) {
    const got = new Tandem(FIXTURE_KEY, { position: BigInt(pos) }).fillExponentialF32(n);
    assertEquals(allNear(near32, got, want.slice(0, n)), true, `f32 from ${pos}`);
  }
});

// The hashes of tandem-c's tools/dump_normals.c, tests/test_normal_bits.c and
// tools/dump_exponentials.c: fills from each of five start positions, seed 2026 + 7 2^64, so
// the hashes cover the slow path of the ziggurat and the whole polynomial range.
const HASH_KEY = seed(2026n + (7n << 64n));
const HASH_STARTS = [0n, 1n, 77n, 12345n, 1n << 30n];
const N = 1_000_000;

test("1e6 Float64 normals at five starts hash to the tandem-c dump", () => {
  const hash = createHash("sha256");
  for (const position of HASH_STARTS) {
    hash.update(bytes(new Tandem(HASH_KEY, { position }).fillNormalF64(N)));
  }
  assertEquals(
    hash.digest("hex"),
    "700ec4d2f4d6b82aaa56c6eff18a4e5919585fdbd093988773383d580ea610d1",
  );
});

test("1e6 exponentials and Float32 normals at five starts stay near the exact forms", () => {
  // f64: against the fused arithmetic of tandem-c on the same uniforms. f32: against libm, in
  // double. The bounds sit a little above the worst deviations measured.
  let e64 = 0, e32 = 0, n32 = 0;
  for (const position of HASH_STARTS) {
    const u64 = new Tandem(HASH_KEY, { position }).fillF64(N);
    const x64 = new Tandem(HASH_KEY, { position }).fillExponentialF64(N);
    for (let i = 0; i < N; i++) {
      const want = 0.5 * neg2Log64(1 - u64[i], true);
      e64 = Math.max(e64, Math.abs(x64[i] - want) / (2 ** -52 * want));
    }
    const u32 = new Tandem(HASH_KEY, { position }).fillF32(2 * N);
    const x32 = new Tandem(HASH_KEY, { position }).fillExponentialF32(2 * N);
    const z32 = new Tandem(HASH_KEY, { position }).fillNormalF32(2 * N);
    for (let i = 0; i < 2 * N; i++) {
      const want = -Math.log1p(-u32[i]);
      e32 = Math.max(e32, Math.abs(x32[i] - want) / (2 ** -23 * want));
      const j = i & ~1, r = Math.sqrt(-2 * Math.log1p(-u32[j])), b = 2 * Math.PI * u32[j + 1];
      n32 = Math.max(n32, Math.abs(z32[i] - (i & 1 ? r * Math.sin(b) : r * Math.cos(b))));
    }
  }
  console.log(`worst: exponential f64 ${e64} ulps, f32 ${e32} ulps, normal f32 ${n32}`);
  assertEquals(e64 < 4 && e32 < 3 && n32 < 1e-6, true);
});

// erfc with fractional error under 1.2e-7 (Numerical Recipes' erfcc), far below the KS bound.
const ERFC = [
  -1.26551223,
  1.00002368,
  0.37409196,
  0.09678418,
  -0.18628806,
  0.27886807,
  -1.13520398,
  1.48851587,
  -0.82215223,
  0.17087277,
];
function erfc(x: number): number {
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  let p = 0;
  for (let k = ERFC.length - 1; k >= 0; k--) p = ERFC[k] + t * p;
  const r = t * Math.exp(-z * z + p);
  return x >= 0 ? r : 2 - r;
}

// Raw moments 1 to 4 and E[X^2k], whose difference with the squared moment is the variance.
type Law = { m: number[]; m2k: number[]; cdf: (x: number) => number };
const EXP1: Law = { m: [1, 2, 6, 24], m2k: [2, 24, 720, 40320], cdf: (x) => -Math.expm1(-x) };
const NORMAL: Law = {
  m: [0, 1, 0, 3],
  m2k: [1, 3, 15, 105],
  cdf: (x) => 0.5 * erfc(-x / Math.SQRT2),
};
const LAW_N = 10_000_000;

// One test per law and precision, so each stays under Bun's 5 s default timeout.
for (
  const [name, law, draw] of [
    ["exponential f64", EXP1, (g: Tandem) => g.fillExponentialF64(LAW_N)],
    ["exponential f32", EXP1, (g: Tandem) => g.fillExponentialF32(LAW_N)],
    ["normal f64", NORMAL, (g: Tandem) => g.fillNormalF64(LAW_N)],
    ["normal f32", NORMAL, (g: Tandem) => g.fillNormalF32(LAW_N)],
  ] as const
) {
  test(`1e7 ${name} draws have the first four moments and the KS law`, () => {
    const x = draw(new Tandem(seed(31n))), n = x.length, s = [0, 0, 0, 0];
    for (let i = 0; i < n; i++) {
      const v = x[i], v2 = v * v;
      s[0] += v;
      s[1] += v2;
      s[2] += v2 * v;
      s[3] += v2 * v2;
    }
    for (let k = 0; k < 4; k++) {
      const se = Math.sqrt((law.m2k[k] - law.m[k] ** 2) / n);
      assertEquals(Math.abs(s[k] / n - law.m[k]) < 5 * se, true, `moment ${k + 1}`);
    }
    // Kolmogorov-Smirnov: sqrt(n) D under 1.95 is the 0.1 % critical value.
    const sorted = x.slice().sort();
    let d = 0;
    for (let i = 0; i < n; i++) {
      const c = law.cdf(sorted[i]);
      d = Math.max(d, (i + 1) / n - c, c - i / n);
    }
    assertEquals(Math.sqrt(n) * d < 1.95, true, `KS sqrt(n) D = ${Math.sqrt(n) * d}`);
  });
}

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
  for (const [bits, fma, r] of [[53, fma64, (x: number) => x]] as const) {
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
  // w is s^2 of the logarithm, at most 0.0295.
  for (let i = 0; i < 20000; i++) {
    const w = i % 50 === 0 ? rand() * 2 ** -40 : rand() * 0.0295;
    let acc = LOG_POLY[0];
    for (let k = 1; k < LOG_POLY.length; k++) acc = exactFma(w, acc, LOG_POLY[k], 53);
    assertEquals(horner64(w, LOG_POLY), acc, `w = ${w}`);
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

test("empty fills align the plain kinds and the f64 normals, and leave the rest alone", () => {
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
  assertEquals(run((g) => g.fillNormalF64(0)), 64n);
  for (
    const fill of [
      (g: Tandem) => g.fillU32Below(0, 7),
      (g: Tandem) => g.fillU64Below(0, 7n),
      (g: Tandem) => g.fillNormalF32(0),
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
      n32: fresh().fillNormalF32(300),
      e64: fresh().fillExponentialF64(300),
      e32: fresh().fillExponentialF32(300),
    };
    for (let c = 0; c + 1 < cuts.length; c++) {
      const [lo, hi] = [cuts[c], cuts[c + 1]];
      assertEquals(at(64, lo).fillExponentialF64(hi - lo), w.e64.subarray(lo, hi));
      assertEquals(at(32, lo).fillExponentialF32(hi - lo), w.e32.subarray(lo, hi));
      // A pair is two draws, so a Float32 normal fill cuts at even elements.
      const even = lo - (lo % 2), n = Math.min(hi - even, 300 - even);
      assertEquals(at(32, even).fillNormalF32(n), w.n32.subarray(even, even + n), `K=${K} ${lo}`);
    }
  }
});

test("a Float64 normal fill cut at a miss equals the whole fill and the scalar draws", () => {
  const key = seed(21n), start = 37n, p = align(start, 64), n = 3000;
  for (const K of [32, 8]) {
    // The draws that miss the inner rectangles and take the fallback generator.
    const raw = new Tandem(key, { position: start, K }).fillU64(n);
    const misses = [...raw.keys()].filter((i) =>
      raw[i] >> 11n >= BigInt(ZIG_K[Number(raw[i] & 1023n)])
    );
    assertEquals(misses.length >= 3, true);
    const whole = new Tandem(key, { position: start, K }).fillNormalF64(n);
    const cuts = [0, misses[0], misses[0] + 1, misses[2], n];
    for (let c = 0; c + 1 < cuts.length; c++) {
      const [a, b] = [cuts[c], cuts[c + 1]];
      const g = new Tandem(key, { position: p + 64n * BigInt(a), K });
      assertEquals(g.fillNormalF64(b - a), whole.subarray(a, b), `K=${K} [${a},${b})`);
    }
    const one = new Tandem(key, { position: start, K });
    assertEquals(Float64Array.from({ length: n }, () => one.nextNormalF64()), whole);
    assertEquals(one.position, p + 64n * BigInt(n));
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
  assertEquals(z.position, align(position, 64) + 64n * 51n);
  const e = fillCpu({ key, position, count: 51, dtype: "f32", exponential: true });
  assertEquals(e.values, new Tandem(key, { position }).fillExponentialF32(51));
});

test("the JavaScript kernel gives the words of the WebAssembly kernel", () => {
  // Fills across chunk groups and the 256-row output buffer, and scalar draws through windows.
  const run = () =>
    [1, 8, 32].flatMap((K) => {
      const g = new Tandem(seed(9n), { position: 77n, K });
      return [g.fillU32(300 * 32 + 5), Uint32Array.from({ length: 700 }, () => g.nextU32())];
    });
  const hasWasm = useWasm(true), wasm = run();
  useWasm(false);
  try {
    assertEquals(run(), wasm);
  } finally {
    useWasm(true);
  }
  console.log(`WebAssembly kernel ${hasWasm ? "in use" : "missing, both runs used JavaScript"}`);
});
