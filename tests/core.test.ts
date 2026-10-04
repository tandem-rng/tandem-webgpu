// The CPU building blocks against the spec vectors.
import { assertEquals, assertThrows } from "jsr:@std/assert@1";
import {
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
import vectors from "./vectors.json" with { type: "json" };

const DOMAIN_STREAM = 0x9e3779b9, AUX_STREAM = 0x94d049bb;
const words = (ws: string[]) =>
  ws.map((w) => parseInt(w, 16) >>> 0) as [number, number, number, number];
const KEY = words(vectors.key);

Deno.test("T", () => {
  for (const v of vectors.T) {
    const s = T({ o: words(v.o), h: words(v.h) });
    assertEquals(s.o, words(v.o_out));
    assertEquals(s.h, words(v.h_out));
  }
});

Deno.test("F", () => {
  for (const v of vectors.F) {
    const s = fKeyed(KEY, BigInt(v.counter), DOMAIN_STREAM, AUX_STREAM);
    assertEquals(s.o, words(v.o));
    assertEquals(s.h, words(v.h));
  }
  assertEquals(F({ o: [0, 0, 0, 0], h: [0, 0, 0, 0] }).o.length, 4);
});

Deno.test("stream words via block", () => {
  for (const s of vectors.stream_words) {
    const row = BigInt(s.row), K = BigInt(vectors.K);
    assertEquals(block(KEY, 8n * (row / K) + BigInt(s.lane), Number(row % K)), words(s.words));
  }
});

Deno.test("draws from position 0", () => {
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

Deno.test("derived keys", () => {
  const k = vectors.derived_keys;
  assertEquals(split(KEY, 0n), words(k.split_child_0));
  assertEquals(split(KEY, 1n), words(k.split_child_1));
  assertEquals(sub(KEY, 7n), words(k.purpose_7));
  const f = fork(KEY, 0n, 1);
  assertEquals(f.children[0], words(k.fork_child_0_at_block_0));
  assertEquals(f.position, 128n);
});

Deno.test("seed whitening", () => {
  const s = vectors.seed_whitening;
  assertEquals(seed(BigInt(s.seed)), words(s.key));
});

const dump = (name: string) => Deno.readFile(new URL(`./data/${name}`, import.meta.url));
const K1234: [number, number, number, number] = [1, 2, 3, 4];

Deno.test("Tandem draws from position 0 match the spec vectors", () => {
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

Deno.test("Tandem fills equal the Julia dumps at K = 32 and K = 8", async () => {
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

Deno.test("Tandem scalar draws interleave widths with alignment", async () => {
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

Deno.test("Tandem fills from mid-stream positions across chunk groups", async () => {
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

Deno.test("Tandem derived generators match the spec vectors", () => {
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

Deno.test("Tandem enforces the 2^64 position bound", () => {
  const g = new Tandem(KEY, { position: (1n << 64n) - 32n });
  assertThrows(() => g.nextU32(), RangeError);
  assertEquals(g.fillU32(0).length, 0);
});
