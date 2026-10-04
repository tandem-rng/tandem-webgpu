// The CPU building blocks against the spec vectors.
import {
  align,
  block,
  F,
  fKeyed,
  fork,
  seed,
  split,
  sub,
  T,
  Tandem,
  toFloat32,
  toFloat64,
} from "../src/mod.ts";
import { assertEquals, assertThrows, dump, test } from "./harness.ts";
import vectors from "./vectors.json" with { type: "json" };

const DOMAIN_STREAM = 0x9e3779b9, AUX_STREAM = 0x94d049bb;
const words = (ws: string[]) =>
  ws.map((w) => parseInt(w, 16) >>> 0) as [number, number, number, number];
const KEY = words(vectors.key);

test("T", () => {
  for (const v of vectors.T) {
    const s = T({ o: words(v.o), h: words(v.h) });
    assertEquals(s.o, words(v.o_out));
    assertEquals(s.h, words(v.h_out));
  }
});

test("F", () => {
  for (const v of vectors.F) {
    const s = fKeyed(KEY, BigInt(v.counter), DOMAIN_STREAM, AUX_STREAM);
    assertEquals(s.o, words(v.o));
    assertEquals(s.h, words(v.h));
  }
  assertEquals(F({ o: [0, 0, 0, 0], h: [0, 0, 0, 0] }).o.length, 4);
});

test("stream words via block", () => {
  for (const s of vectors.stream_words) {
    const row = BigInt(s.row), K = BigInt(vectors.K);
    assertEquals(block(KEY, 8n * (row / K) + BigInt(s.lane), Number(row % K)), words(s.words));
  }
});

test("draws from position 0", () => {
  const d = vectors.draws_from_position_0;
  // Word i sits in row i >> 5, lane (i >> 2) & 7, word i & 3 of that block.
  const word = (i: number) => {
    const row = i >> 5;
    return block(KEY, 8n * BigInt(row >> 5) + BigInt((i >> 2) & 7), row & 31)[i & 3];
  };
  for (const [i, x] of Object.entries(d.Float64)) {
    const k = Number(i);
    const raw = BigInt(word(2 * k)) | (BigInt(word(2 * k + 1)) << 32n);
    assertEquals(toFloat64(raw), x);
  }
  for (const [i, x] of Object.entries(d.Float32)) {
    assertEquals(toFloat32(word(Number(i))), Math.fround(x));
  }
  for (const [i, x] of Object.entries(d.Bool)) {
    const k = Number(i);
    assertEquals((word(k >> 5) >>> (k & 31)) & 1, x);
  }
});

test("derived keys", () => {
  const k = vectors.derived_keys;
  assertEquals(split(KEY, 0n), words(k.split_child_0));
  assertEquals(split(KEY, 1n), words(k.split_child_1));
  assertEquals(sub(KEY, 7n), words(k.purpose_7));
  const f = fork(KEY, 0n, 1);
  assertEquals(f.children[0], words(k.fork_child_0_at_block_0));
  assertEquals(f.position, 128n);
});

test("seed whitening", () => {
  const s = vectors.seed_whitening;
  assertEquals(seed(BigInt(s.seed)), words(s.key));
});

const K1234: [number, number, number, number] = [1, 2, 3, 4];

test("Tandem draws from position 0 match the spec vectors", () => {
  const d = vectors.draws_from_position_0;
  for (const [i, x] of Object.entries(d.Float64)) {
    assertEquals(new Tandem(KEY).atF64(Number(i)), x);
  }
  for (const [i, x] of Object.entries(d.Float32)) {
    assertEquals(new Tandem(KEY).atF32(Number(i)), Math.fround(x));
  }
  const g = new Tandem(KEY);
  const bits = Array.from({ length: 129 }, () => g.nextBool());
  for (const [i, x] of Object.entries(d.Bool)) assertEquals(bits[Number(i)], x === 1);
  assertEquals(g.position, 129n);
});

test("Tandem fills equal the Julia dumps at K = 32 and K = 8", async () => {
  const u32 = new Uint32Array((await dump("k1234_K32_u32.bin")).buffer);
  assertEquals(new Tandem(K1234).fillU32(u32.length), u32);
  const k8 = new Uint32Array((await dump("k1234_K8_u32.bin")).buffer);
  assertEquals(new Tandem(K1234, { K: 8 }).fillU32(k8.length), k8);
  const u64 = new BigUint64Array((await dump("k1234_K32_u64.bin")).buffer);
  assertEquals(new Tandem(K1234).fillU64(u64.length), u64);
  const f32 = new Float32Array((await dump("seed42_K32_f32.bin")).buffer);
  assertEquals(Tandem.seed(42n).fillF32(f32.length), f32);
  const f64 = new Float64Array((await dump("seed42_K32_f64.bin")).buffer);
  assertEquals(Tandem.seed(42n).fillF64(f64.length), f64);
});

test("Tandem scalar draws interleave widths with alignment", async () => {
  const u8 = new Uint8Array(await dump("seed42_K32_u8.bin"));
  const g = Tandem.seed(42n);
  assertEquals(g.nextU8(), u8[0]);
  assertEquals(g.nextBool(), (u8[1] & 1) === 1);
  assertEquals(g.position, 9n);
  // The u16 aligns to bit 16, the u32 to bit 32 and the u64 to bit 64.
  assertEquals(g.nextU16(), u8[2] | (u8[3] << 8));
  assertEquals(g.nextU32(), new DataView(u8.buffer).getUint32(4, true));
  assertEquals(g.nextU64(), new DataView(u8.buffer).getBigUint64(8, true));
  assertEquals(g.position, 128n);
});

test("Tandem fills from mid-stream positions across chunk groups", async () => {
  for (const [name, K] of [["k1234_K32_u32.bin", 32], ["k1234_K8_u32.bin", 8]] as const) {
    const want = new Uint32Array((await dump(name)).buffer);
    for (const start of [1, 31, 1025, 4097]) {
      const g = new Tandem(K1234, { position: BigInt(32 * start), K });
      assertEquals(g.fillU32(300), want.subarray(start, start + 300), `${name} from ${start}`);
      assertEquals(g.position, BigInt(32 * (start + 300)));
      assertEquals(
        new Tandem(K1234, { position: BigInt(32 * start), K }).atU32(299),
        want[start + 299],
      );
    }
  }
});

test("Tandem derived generators match the spec vectors", () => {
  const k = vectors.derived_keys, g = new Tandem(KEY, { K: 8 });
  assertEquals(g.split(0).key, words(k.split_child_0));
  assertEquals(g.split(1).key, words(k.split_child_1));
  assertEquals(g.sub(7).key, words(k.purpose_7));
  assertEquals(g.split(0).chunkLength, 8);
  assertEquals(g.split(0).position, 0n);
  g.nextU32();
  const [child] = g.fork(1);
  assertEquals(child.key, words(k.fork_child_0_at_block_0));
  assertEquals(g.position, 128n);
  assertEquals(
    Tandem.seed(BigInt(vectors.seed_whitening.seed)).key,
    words(vectors.seed_whitening.key),
  );
});

test("Tandem enforces the 2^64 position bound", () => {
  const g = new Tandem(KEY, { position: (1n << 64n) - 32n });
  assertThrows(() => g.nextU32(), RangeError);
  assertEquals(g.fillU32(0).length, 0);
});

// Fixtures of tandem-c and tandem-cuda (tools/gen_cross.ts). Structs are positional arrays and
// 64-bit integers are decimal strings.
import cross from "./cross.json" with { type: "json" };

type Below = [string, string[], string]; // range, 64 values, end position
type DeviceBelow = [string, number, string[]]; // range, rejected count, 64 values
type CFill = [string, string, string[], string]; // start position, range, 64 values, end position
type DeviceNormal = [string, number, number[]]; // start position, n, 64 values
const FIXTURE_KEY = cross.CROSS_FILL_KEY as [number, number, number, number];

test("scalar bounded draws match tandem-c, positions included", () => {
  for (const [range, want, end] of cross.CROSS_U32 as unknown as Below[]) {
    const g = Tandem.seed(42n);
    g.nextBool();
    assertEquals(want.map(() => g.nextU32Below(Number(range))), want.map(Number));
    assertEquals(g.position, BigInt(end), `u32 range ${range}`);
  }
  for (const [range, want, end] of cross.CROSS_U64 as unknown as Below[]) {
    const g = Tandem.seed(42n);
    g.nextBool();
    assertEquals(want.map(() => g.nextU64Below(BigInt(range))), want.map(BigInt));
    assertEquals(g.position, BigInt(end), `u64 range ${range}`);
  }
});

test("bounded fills from position 0 match tandem-cuda, rejections included", () => {
  let rejected = 0;
  for (const [range, rej, want] of cross.CROSS_BELOW32 as unknown as DeviceBelow[]) {
    assertEquals(
      new Tandem(FIXTURE_KEY).fillU32Below(64, Number(range)),
      Uint32Array.from(want, Number),
    );
    rejected += rej;
  }
  for (const [range, rej, want] of cross.CROSS_BELOW64 as unknown as DeviceBelow[]) {
    assertEquals(
      new Tandem(FIXTURE_KEY).fillU64Below(64, BigInt(range)),
      BigUint64Array.from(want, BigInt),
    );
    rejected += rej;
  }
  assertEquals(rejected > 0, true);
});

test("bounded fills from positions 1 and 12345 match tandem-c", () => {
  for (const [start, range, want, end] of cross.CROSS_FILL_U32 as unknown as CFill[]) {
    const g = new Tandem(seed(42n), { position: BigInt(start) });
    assertEquals(g.fillU32Below(64, Number(range)), Uint32Array.from(want, Number));
    assertEquals(g.position, BigInt(end), `u32 from ${start} range ${range}`);
  }
  for (const [start, range, want, end] of cross.CROSS_FILL_U64 as unknown as CFill[]) {
    const g = new Tandem(seed(42n), { position: BigInt(start) });
    assertEquals(g.fillU64Below(64, BigInt(range)), BigUint64Array.from(want, BigInt));
    assertEquals(g.position, BigInt(end), `u64 from ${start} range ${range}`);
  }
});

test("bounded range 0 gives 0 and consumes a draw, an empty fill moves nothing", () => {
  const g = Tandem.seed(5n);
  assertEquals(g.fillU32Below(3, 0), new Uint32Array(3));
  assertEquals(g.position, 96n);
  assertEquals(g.nextU64Below(0n), 0n);
  assertEquals(g.position, 192n);
  const h = new Tandem(KEY, { position: 5n });
  h.fillU32Below(0, 7);
  h.fillU64Below(0, 7n);
  assertEquals(h.position, 5n);
});

/** Draws of the plain fill that Lemire rejects, so a test knows the fallback ran. */
function rejections(raw: ArrayLike<number | bigint>, range: bigint, bits: bigint): number {
  const t = ((1n << bits) - range) % range;
  return Array.from(raw).filter((x) => ((BigInt(x) * range) & ((1n << bits) - 1n)) < t).length;
}

test("a bounded fill cut at any boundary equals the whole fill", () => {
  const key = seed(11n), start = 37n, cuts = [0, 101, 233, 300];
  for (const K of [32, 8]) {
    const u32 = 2147483649, p32 = align(start, 32);
    const whole32 = new Tandem(key, { position: start, K }).fillU32Below(300, u32);
    assertEquals(
      rejections(new Tandem(key, { position: start, K }).fillU32(300), BigInt(u32), 32n) > 0,
      true,
    );
    const u64 = 9223372036854775809n, p64 = align(start, 64);
    const whole64 = new Tandem(key, { position: start, K }).fillU64Below(300, u64);
    assertEquals(
      rejections(new Tandem(key, { position: start, K }).fillU64(300), u64, 64n) > 0,
      true,
    );
    for (let c = 0; c + 1 < cuts.length; c++) {
      const [a, b] = [cuts[c], cuts[c + 1]];
      const p32a = new Tandem(key, { position: p32 + 32n * BigInt(a), K });
      assertEquals(p32a.fillU32Below(b - a, u32), whole32.subarray(a, b), `u32 K=${K} [${a},${b})`);
      const p64a = new Tandem(key, { position: p64 + 64n * BigInt(a), K });
      assertEquals(p64a.fillU64Below(b - a, u64), whole64.subarray(a, b), `u64 K=${K} [${a},${b})`);
    }
  }
});

test("normals equal tandem-c bit for bit from an unaligned start", () => {
  const f64 = new Tandem(seed(42n));
  f64.nextBool();
  const want64 = Float64Array.from(cross.CROSS_NORMAL as number[]);
  assertEquals(f64.fillNormalF64(want64.length), want64);
  assertEquals(f64.position, BigInt(cross.CROSS_NORMAL_END_POS));
  const f32 = new Tandem(seed(42n));
  f32.nextBool();
  const want32 = Float32Array.from(cross.CROSS_NORMALF as number[]);
  assertEquals(f32.fillNormalF32(want32.length), want32);
  assertEquals(f32.position, BigInt(cross.CROSS_NORMALF_END_POS));
});

test("normals equal tandem-cuda fills bit for bit at several positions, odd counts included", () => {
  for (const [pos, n, want] of cross.CROSS_NORMAL64 as unknown as DeviceNormal[]) {
    const got = new Tandem(FIXTURE_KEY, { position: BigInt(pos) }).fillNormalF64(n);
    assertEquals(got, Float64Array.from(want.slice(0, n)), `f64 from ${pos}`);
  }
  for (const [pos, n, want] of cross.CROSS_NORMAL32 as unknown as DeviceNormal[]) {
    const g = new Tandem(FIXTURE_KEY, { position: BigInt(pos) });
    assertEquals(g.fillNormalF32(n), Float32Array.from(want.slice(0, n)), `f32 from ${pos}`);
    assertEquals(g.position, align(BigInt(pos), 32) + 32n * BigInt(n + (n % 2)));
  }
});

test("a scalar normal is the cosine half of the pair and consumes two draws", () => {
  const g = Tandem.seed(8n), h = Tandem.seed(8n);
  assertEquals(g.nextNormalF64(), h.fillNormalF64(1)[0]);
  assertEquals(g.position, 128n);
  assertEquals(g.nextNormalF32(), h.fillNormalF32(2)[0]);
  assertEquals(new Tandem(KEY, { position: 5n }).fillNormalF32(0).length, 0);
});
