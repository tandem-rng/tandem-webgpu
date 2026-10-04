<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .wgsl"></p>

# tandem-webgpu

[Tandem8x32](https://github.com/tandem-rng/spec) for WebGPU: a noncryptographic pseudorandom
number generator built to be fast on CPUs and GPUs alike. A WGSL compute shader fills the
stream on the GPU, and a small TypeScript module derives keys and small draws on the CPU. It
produces the stream the specification defines, bit for bit.

- `tandem.wgsl`: the building blocks `T`, `F`, `F_keyed`, `block`, `split_key`, `sub_key`,
  and the `fill` entry point. WGSL has no 64-bit integers, so the 32x32 to 64 product of the
  mix is built from 16-bit halves, which is exact, and chunk indices travel as word pairs.
- `tandem_f32.wgsl`: the `fill_f32` entry point, which stores Float32 values instead of words.
  It is appended to `tandem.wgsl` when the shader is embedded, so `tandem.wgsl` stays identical
  to the copy in tandem-rs.
- `src/`: the `Tandem` class for draws and fills on the CPU, the key functions `seed`, `split`,
  `sub`, `fork`, the float mappings, and `fill`, which runs the shader and reads back a typed
  array. No dependencies.

## Use

```ts
import { fill, fillBuffer, fork, requestDevice, seed, split, Tandem } from "tandem-webgpu";

const device = await requestDevice();          // a device with the adapter's buffer limits
const key = seed(42n);                          // 128-bit seed through the spec's whitening
const { values, position } = await fill(device, { key, count: 1 << 20, dtype: "f64" });
const next = await fill(device, { key, position, count: 1000, dtype: "u32" });
const worker = split(key, 7n);                  // by index, from the key alone
const { children } = fork(key, position, 4);    // from the current block
const { buffer } = await fillBuffer(device, { key, count: 1 << 24, dtype: "u32" }); // stays on the GPU
const floats = await fillBuffer(device, { key, count: 1 << 24, dtype: "f32", floats: true }); // f32 values

const rng = Tandem.seed(42n);                   // small draws on the CPU, no device needed
rng.nextF64(); rng.nextU8(); rng.nextBool();    // each aligns to its width, as the spec says
const xs = rng.fillF32(1000);                   // also fillU32, fillU64, fillF64
rng.atU32(5n);                                  // element 5 of the next fill, position unmoved
const child = rng.split(7);                     // also sub(purpose) and fork(n), as Tandem objects
```

`dtype` is one of `u8`, `u16`, `u32`, `u64`, `f32`, `f64`. The `fill` entry point writes stream
words. `fill` of `f32` runs `fill_f32`, which applies the spec's mapping `(raw >> 8) * 2^-24` on
the GPU, and `fill` of `f64` applies `(raw >> 11) * 2^-53` on the host. `fillBuffer` returns a
storage buffer of whole 16-byte stream blocks plus the byte offset of the first value, holding
raw words unless you pass `floats: true` with `dtype: "f32"`, which stores Float32 values for a
later GPU stage. Pass `buffer` to write into your own. There is no `f64` on the GPU: WGSL has
no 64-bit float type, so `f64` stays a host mapping of the `u64` words.

`Tandem` takes a key and an optional `position` and `K`, and exposes `key`, `position` and
`chunkLength`. Its draws are the spec's scalar draws: `nextU8`, `nextU16`, `nextU32`, `nextU64`,
`nextF32`, `nextF64` and `nextBool`. Each draw costs one `block` call per 16 bytes, so use
`fill` on the GPU for bulk output.

Deno runs the TypeScript directly. For browsers and Node, `npm run build` emits `dist/`.

## Tests

```sh
deno task test
```

`tests/core.test.ts` checks the CPU building blocks against every vector of the specification
(`tests/vectors.json`, a copy of the spec repository's file). It also checks the `Tandem` class:
its fills against every dump in `tests/data` at K = 32 and K = 8, its mixed-width draws, its
fills from mid-stream positions, its derived generators, and the 2^64 position bound. `tests/gpu.test.ts` checks the
GPU fill against the vectors and against reference stream dumps in `tests/data`
for u32 at K = 32 and K = 8, u64, f32, f64 and u8, from several start positions
and across workgroup boundaries. It also checks that `fillBuffer` with `floats` holds the
mapped values and that without it the buffer keeps raw words. The GPU tests skip when no adapter exists. CI runs them with
a software Vulkan adapter on Linux and on the macOS runner's GPU, validates the shader with
`naga`, and fails when the embedded shader or the vectors drift.

## Speed

The fill is compute bound on Apple GPUs at about 160 to 210 GiB/s, under Chromium's Tint and
under Deno's naga alike. The store ceiling of the same buffer and dispatch
shape is 720 GiB/s. A submit and its completion cost about 0.4 ms on wgpu, so a single small
fill is latency bound: batch fills in one submit or fill large buffers.

Apple M4 Pro GPU, `fill_u32` into device memory with no readback, minimum of 7 after a
warm-up:

| | GiB/s |
|---|---|
| Chromium (Dawn, Tint), 2^26 and 2^27 words, one fill per submit | 167 |
| Chromium, 2^24 words, 16 fills per submit | 213 |
| Deno 2.9 (wgpu, naga), 2^26 words, 32 fills per submit | 134 to 144 |
| Deno 2.9, 2^26 words, one fill per submit | 18.5 |
| store ceiling: one constant block per invocation, same buffer, 32 per submit | 700 to 720 |

Variants measured on this GPU and rejected, all within noise of the committed shader or
slower: a plain `*` in place of the exact 16-bit-half `mul_hi`, no bounds check, workgroups
of 64 or 128, four steps unrolled per store burst, a lane-major thread mapping, two chunks
per invocation, and a workgroup tile with 512-byte writes per SIMD group, which
measured four times slower. `deno task bench` prints the Deno rows and the ceiling.

`demo/index.html` runs the fill in a browser: `python3 -m http.server` in the repo root and
open `/demo/`.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
