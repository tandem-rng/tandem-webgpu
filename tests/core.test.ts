// The CPU building blocks against the spec vectors.
import { assertEquals } from "jsr:@std/assert@1";
import { block, F, fKeyed, fork, seed, split, sub, T, toFloat32, toFloat64 } from "../src/mod.ts";
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
