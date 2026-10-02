// The GPU fill against the Julia dumps and the spec vectors. Skipped without an adapter.
import { assertEquals } from "jsr:@std/assert@1";
import { fill, requestDevice, seed } from "../src/mod.ts";
import vectors from "./vectors.json" with { type: "json" };

const adapter = await navigator.gpu?.requestAdapter();
const device = adapter ? await requestDevice(adapter) : null;
const gpu = { ignore: !device };
const KEY = vectors.key.map((w) => parseInt(w, 16) >>> 0) as [number, number, number, number];
const K1234 = [1, 2, 3, 4] as const;

const dump = (name: string) => Deno.readFile(new URL(`./data/${name}`, import.meta.url));

Deno.test({ name: "stream words from the vectors", ...gpu }, async () => {
  const { values } = await fill(device!, { key: KEY, count: 64, dtype: "u32" });
  for (const s of vectors.stream_words) {
    assertEquals(
      [...values.subarray(s.first_word, s.first_word + 4)],
      s.words.map((w) => parseInt(w, 16) >>> 0),
    );
  }
});

Deno.test({ name: "u32 dump, K = 32 and K = 8, from several offsets", ...gpu }, async () => {
  for (const [name, K] of [["k1234_K32_u32.bin", 32], ["k1234_K8_u32.bin", 8]] as const) {
    const want = new Uint32Array((await dump(name)).buffer);
    for (const start of [0, 1, 7, 31, 32, 33, 257, 1024, 4097]) {
      const { values, position } = await fill(device!, {
        key: K1234,
        position: BigInt(32 * start),
        count: want.length - start,
        dtype: "u32",
        K,
      });
      assertEquals(values, want.subarray(start), `${name} from ${start}`);
      assertEquals(position, BigInt(32 * want.length));
    }
  }
});

Deno.test({ name: "u64 dump", ...gpu }, async () => {
  const want = new BigUint64Array((await dump("k1234_K32_u64.bin")).buffer);
  const { values } = await fill(device!, { key: K1234, count: want.length, dtype: "u64" });
  assertEquals(values, want);
});

Deno.test({ name: "f32 and f64 dumps from seed 42", ...gpu }, async () => {
  const key = seed(42n);
  const f32 = new Float32Array((await dump("seed42_K32_f32.bin")).buffer);
  assertEquals((await fill(device!, { key, count: f32.length, dtype: "f32" })).values, f32);
  const f64 = new Float64Array((await dump("seed42_K32_f64.bin")).buffer);
  assertEquals((await fill(device!, { key, count: f64.length, dtype: "f64" })).values, f64);
});

Deno.test({ name: "u8 dump and mid-word positions", ...gpu }, async () => {
  const want = new Uint8Array(await dump("seed42_K32_u8.bin"));
  const key = seed(42n);
  assertEquals((await fill(device!, { key, count: want.length, dtype: "u8" })).values, want);
  const { values, position } = await fill(device!, {
    key,
    position: 8n * 3n,
    count: 100,
    dtype: "u8",
  });
  assertEquals(values, want.subarray(3, 103));
  assertEquals(position, 8n * 103n);
});

Deno.test({ name: "alignment: a u16 after an odd bit position", ...gpu }, async () => {
  const key = seed(42n);
  const want = new Uint8Array(await dump("seed42_K32_u8.bin"));
  const { values, position } = await fill(device!, { key, position: 9n, count: 2, dtype: "u16" });
  assertEquals(position, 48n);
  assertEquals(values[0], want[2] | (want[3] << 8));
  assertEquals(values[1], want[4] | (want[5] << 8));
});

Deno.test({ name: "a fill that spans many workgroups and groups", ...gpu }, async () => {
  const key = seed(7n);
  const n = 1 << 20;
  const whole = (await fill(device!, { key, count: n, dtype: "u32" })).values;
  const start = 300_001;
  const part =
    (await fill(device!, { key, position: 32n * BigInt(start), count: n - start, dtype: "u32" }))
      .values;
  assertEquals(part, whole.subarray(start));
  assertEquals(whole.length, n);
});
