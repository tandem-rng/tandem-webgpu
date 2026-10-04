// The GPU fill against the Julia dumps and the spec vectors. Skipped without an adapter.
import { assertEquals } from "jsr:@std/assert@1";
import { fill, fillBuffer, fillMany, requestDevice, seed, Tandem, toFloat32 } from "../src/mod.ts";
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

async function readBytes(buffer: GPUBuffer, offset: number, length: number): Promise<ArrayBuffer> {
  const staging = device!.createBuffer({ size: buffer.size, usage: 0x1 | 0x8 });
  const encoder = device!.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
  device!.queue.submit([encoder.finish()]);
  await staging.mapAsync(0x1);
  const bytes = staging.getMappedRange().slice(offset, offset + length);
  staging.destroy();
  return bytes;
}

Deno.test({ name: "fillBuffer floats: GPU-resident f32, mid-block start", ...gpu }, async () => {
  const key = seed(42n), position = 32n * 5n, count = 1000;
  const raw = (await fill(device!, { key, position, count, dtype: "u32" })).values;
  const r = await fillBuffer(device!, { key, position, count, dtype: "f32", floats: true });
  const floats = new Float32Array(await readBytes(r.buffer, r.byteOffset, r.byteLength));
  assertEquals(floats, Float32Array.from(raw, toFloat32));
  // Without the option the buffer keeps raw words.
  const plain = await fillBuffer(device!, { key, position, count, dtype: "f32" });
  const words = new Uint32Array(await readBytes(plain.buffer, plain.byteOffset, plain.byteLength));
  assertEquals(words, raw);
});

Deno.test(
  { name: "fillMany equals single fills, with mixed dtypes, K and buffers", ...gpu },
  async () => {
    const key = seed(9n);
    const own = device!.createBuffer({ size: 4096, usage: 0x80 | 0x4 });
    const items = [
      { key, count: 1000, dtype: "u32" },
      { key, position: 7n, count: 300, dtype: "u64", K: 8 },
      { key: K1234, position: 3200n, count: 500, dtype: "f32", floats: true },
      { key, position: 5n, count: 100, dtype: "u8", buffer: own },
    ] as const;
    const many = await fillMany(device!, items);
    const types = [Uint32Array, BigUint64Array, Float32Array, Uint8Array];
    for (const [i, item] of items.entries()) {
      const { values, position } = await fill(device!, item);
      const got = new types[i](
        await readBytes(many[i].buffer, many[i].byteOffset, many[i].byteLength),
      );
      assertEquals(got, values, `item ${i}`);
      assertEquals(many[i].position, position);
    }
    assertEquals(many[3].buffer, own);
  },
);

Deno.test(
  { name: "signed integers are the dump bytes read two's complement", ...gpu },
  async () => {
    const u8 = await dump("seed42_K32_u8.bin");
    const key = seed(42n), n = 100;
    const bytes = u8.slice(0, 8 * n);
    const want = {
      i8: new Int8Array(bytes.buffer, 0, n),
      i16: new Int16Array(bytes.buffer, 0, n),
      i32: new Int32Array(bytes.buffer, 0, n),
      i64: new BigInt64Array(bytes.buffer, 0, n),
    };
    for (const dtype of ["i8", "i16", "i32", "i64"] as const) {
      const { values, position } = await fill(device!, { key, count: n, dtype });
      assertEquals(values, want[dtype], dtype);
      assertEquals(position, BigInt(want[dtype].BYTES_PER_ELEMENT * 8 * n));
    }
  },
);

Deno.test(
  { name: "bool: spec vector bits and a mid-word start across blocks", ...gpu },
  async () => {
    for (const [i, x] of Object.entries(vectors.draws_from_position_0.Bool)) {
      const { values } = await fill(device!, {
        key: KEY,
        position: BigInt(i),
        count: 1,
        dtype: "bool",
      });
      assertEquals(values[0], x);
    }
    const key = seed(42n), position = 37n, count = 500;
    const { values, position: next } = await fill(device!, { key, position, count, dtype: "bool" });
    const cpu = new Tandem(key, { position });
    assertEquals(values, Uint8Array.from({ length: count }, () => cpu.nextBool() ? 1 : 0));
    assertEquals(next, position + BigInt(count));
  },
);

Deno.test({ name: "fill leaves a caller buffer alive for reuse", ...gpu }, async () => {
  const key = seed(3n), buffer = device!.createBuffer({ size: 4096, usage: 0x80 | 0x4 });
  const a = await fill(device!, { key, count: 100, dtype: "u32", buffer });
  const b = await fill(device!, { key, position: a.position, count: 100, dtype: "u32", buffer });
  const whole = (await fill(device!, { key, count: 200, dtype: "u32" })).values;
  assertEquals(a.values, whole.subarray(0, 100));
  assertEquals(b.values, whole.subarray(100));
  buffer.destroy();
});
