// The GPU fill against the Julia dumps and the spec vectors. Skipped without an adapter.
import { assertEquals } from "jsr:@std/assert@1";
import {
  align,
  fill,
  fillBelow,
  fillBuffer,
  fillMany,
  fillNormal,
  requestDevice,
  seed,
  Tandem,
  toFloat32,
} from "../src/mod.ts";
import cross from "./cross.json" with { type: "json" };
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

type DeviceBelow = [string, number, string[]];
type CFill = [string, string, string[], string];
type DeviceNormal = [string, number, number[]];
const FIXTURE_KEY = cross.CROSS_FILL_KEY as [number, number, number, number];
const near32 = (got: number, want: number) =>
  Math.abs(got - Math.fround(want)) <= 16 * 2 ** -23 * Math.abs(want) + 1e-6;

Deno.test({ name: "bounded fills equal the tandem-cuda fixtures", ...gpu }, async () => {
  for (const [range, , want] of cross.CROSS_BELOW32 as unknown as DeviceBelow[]) {
    const { values } = await fillBelow(device!, {
      key: FIXTURE_KEY,
      count: 64,
      dtype: "u32",
      range: Number(range),
    });
    assertEquals(values, Uint32Array.from(want, Number), `u32 range ${range}`);
  }
  for (const [range, , want] of cross.CROSS_BELOW64 as unknown as DeviceBelow[]) {
    const { values } = await fillBelow(device!, {
      key: FIXTURE_KEY,
      count: 64,
      dtype: "u64",
      range: BigInt(range),
    });
    assertEquals(values, BigUint64Array.from(want, BigInt), `u64 range ${range}`);
  }
});

Deno.test(
  { name: "bounded fills equal the CPU class with rejections, K and positions", ...gpu },
  async () => {
    const key = seed(21n);
    for (const K of [32, 8]) {
      for (const position of [0n, 37n, 32n * 4099n + 5n]) {
        for (const range of [0, 1, 3, 1000, 2147483649, 4294967295]) {
          const cpu = new Tandem(key, { position, K }).fillU32Below(5000, range);
          const gpuValues = await fillBelow(device!, {
            key,
            position,
            K,
            count: 5000,
            dtype: "u32",
            range,
          });
          assertEquals(gpuValues.values, cpu, `u32 K=${K} from ${position} range ${range}`);
        }
        for (const range of [0n, 1n, 3n, 10n ** 12n, 9223372036854775809n, 18446744073709551615n]) {
          const cpu = new Tandem(key, { position, K }).fillU64Below(3001, range);
          const gpuValues = await fillBelow(device!, {
            key,
            position,
            K,
            count: 3001,
            dtype: "u64",
            range,
          });
          assertEquals(gpuValues.values, cpu, `u64 K=${K} from ${position} range ${range}`);
        }
      }
    }
  },
);

Deno.test(
  { name: "a GPU bounded fill cut at arbitrary boundaries equals the whole fill", ...gpu },
  async () => {
    const key = seed(11n), position = 37n, cuts = [0, 101, 233, 3000];
    const u32 = 2147483649, u64 = 9223372036854775809n;
    const whole32 =
      (await fillBelow(device!, { key, position, count: 3000, dtype: "u32", range: u32 })).values;
    const whole64 =
      (await fillBelow(device!, { key, position, count: 3000, dtype: "u64", range: u64 })).values;
    for (let c = 0; c + 1 < cuts.length; c++) {
      const [a, b] = [cuts[c], cuts[c + 1]];
      const at = (w: bigint) => align(position, Number(w)) + w * BigInt(a);
      const part32 = await fillBelow(device!, {
        key,
        position: at(32n),
        count: b - a,
        dtype: "u32",
        range: u32,
      });
      assertEquals(part32.values, whole32.subarray(a, b), `u32 [${a},${b})`);
      assertEquals(part32.position, align(position, 32) + 32n * BigInt(b));
      const part64 = await fillBelow(device!, {
        key,
        position: at(64n),
        count: b - a,
        dtype: "u64",
        range: u64,
      });
      assertEquals(part64.values, whole64.subarray(a, b), `u64 [${a},${b})`);
    }
  },
);

Deno.test(
  { name: "derived fills batch in fillMany and an empty one leaves the position", ...gpu },
  async () => {
    const key = seed(2n);
    const [a, b, empty, none] = await fillMany(device!, [
      { key, count: 100, dtype: "u32", range: 1000 },
      { key, position: 9n, count: 33, dtype: "f32", normal: true },
      { key, position: 9n, count: 0, dtype: "u32", range: 7 },
      { key, position: 9n, count: 0, dtype: "f32", normal: true },
    ]);
    assertEquals(empty.position, 9n);
    assertEquals(none.position, 9n);
    assertEquals(b.position, 32n + 32n * 34n);
    const cpu = new Tandem(key).fillU32Below(100, 1000);
    assertEquals(new Uint32Array(await readBytes(a.buffer, a.byteOffset, a.byteLength)), cpu);
  },
);

Deno.test(
  { name: "normals match the tandem-cuda fixtures and tandem-c pairs", ...gpu },
  async () => {
    for (const [pos, n, want] of cross.CROSS_NORMAL32 as unknown as DeviceNormal[]) {
      const { values, position } = await fillNormal(device!, {
        key: FIXTURE_KEY,
        position: BigInt(pos),
        count: n,
      });
      assertEquals(values.every((z, i) => near32(z, want[i])), true, `from ${pos}`);
      assertEquals(position, align(BigInt(pos), 32) + 32n * BigInt(n + (n % 2)));
    }
    // tandem-c: seed 42 after a one-bit draw, so the pairs start at an odd stream word.
    const want = cross.CROSS_NORMALF as number[];
    const { values, position } = await fillNormal(device!, {
      key: seed(42n),
      position: 1n,
      count: want.length,
    });
    assertEquals(values.every((z, i) => near32(z, want[i])), true);
    assertEquals(position, BigInt(cross.CROSS_NORMALF_END_POS));
  },
);

Deno.test(
  { name: "normals equal the CPU class across the two-dimensional pair dispatch", ...gpu },
  async () => {
    const key = seed(77n), position = 128n * 13n, n = (1 << 25) - 1;
    const { values } = await fillNormal(device!, { key, position, count: n });
    // The odd count draws 2^25 words, 128 MiB, the smallest binding limit seen (a software
    // adapter), and still needs more pairs than one row of 65535 workgroups holds.
    // Pair 65535 * 256 is the first of the second dispatch row. Check around it and the end.
    for (const start of [2 * 65535 * 256 - 4, 2 * ((n - 1) >> 1) - 2]) {
      const cpu = new Tandem(key, { position: position + 32n * BigInt(start) }).fillNormalF32(
        Math.min(8, n - start),
      );
      assertEquals(
        cpu.every((z, i) => near32(values[start + i], z)),
        true,
        `from element ${start}: got ${values.slice(start, start + 8)} want ${cpu} limit ${
          device!.limits.maxStorageBufferBindingSize
        }`,
      );
    }
    assertEquals(values.length, n);
  },
);
