<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng"></p>

# tandem-webgpu

[Tandem8x32](https://github.com/tandem-rng/spec) for WebGPU: a noncryptographic pseudorandom
number generator built to be fast on CPUs and GPUs alike. A WGSL compute shader fills the
stream on the GPU, and a small TypeScript module derives keys and small draws on the CPU. It
produces the same stream, bit for bit, as [TandemRNG.jl](https://github.com/tandem-rng/TandemRNG.jl),
[tandem-c](https://github.com/tandem-rng/tandem-c), [tandem-rs](https://github.com/tandem-rng/tandem-rs),
[tandem-numpy](https://github.com/tandem-rng/tandem-numpy), [tandem-cuda](https://github.com/tandem-rng/tandem-cuda),
[tandem-jax](https://github.com/tandem-rng/tandem-jax) and [tandem-r](https://github.com/tandem-rng/tandem-r).

- `tandem.wgsl`: the building blocks `T`, `F`, `F_keyed`, `block`, `split_key`, `sub_key`,
  and the `fill` entry point. WGSL has no 64-bit integers, so the 32x32 to 64 product of the
  mix is built from 16-bit halves, which is exact, and chunk indices travel as word pairs.
- `src/`: `seed`, `split`, `sub`, `fork`, the float mappings, and `fill`, which runs the
  shader and reads back a typed array. No dependencies.

## Use

```ts
import { fill, fillBuffer, fork, requestDevice, seed, split } from "tandem-webgpu";

const device = await requestDevice();          // a device with the adapter's buffer limits
const key = seed(42n);                          // 128-bit seed through the spec's whitening
const { values, position } = await fill(device, { key, count: 1 << 20, dtype: "f64" });
const next = await fill(device, { key, position, count: 1000, dtype: "u32" });
const worker = split(key, 7n);                  // by index, from the key alone
const { children } = fork(key, position, 4);    // from the current block
const { buffer } = await fillBuffer(device, { key, count: 1 << 24, dtype: "u32" }); // stays on the GPU
```

`dtype` is one of `u8`, `u16`, `u32`, `u64`, `f32`, `f64`. The shader writes stream words, and
`fill` applies the spec's float mappings on the host: `(raw >> 8) * 2^-24` for `f32` and
`(raw >> 11) * 2^-53` for `f64`. `fillBuffer` returns a storage buffer of whole 16-byte stream
blocks plus the byte offset of the first value; pass `buffer` to write into your own.

Deno runs the TypeScript directly. For browsers and Node, `npm run build` emits `dist/`.

## Tests

```sh
deno task test
```

`tests/core.test.ts` checks the CPU building blocks against every vector of the specification
(`tests/vectors.json`, a copy of the spec repository's file). `tests/gpu.test.ts` checks the
GPU fill against the vectors and against dumps written by TandemRNG.jl (`tests/data`, shared
with tandem-c) for u32 at K = 32 and K = 8, u64, f32, f64 and u8, from several start positions
and across workgroup boundaries. The GPU tests skip when no adapter exists. CI runs them with
a software Vulkan adapter on Linux and on the macOS runner's GPU, validates the shader with
`naga`, and fails when the embedded shader or the vectors drift.

## Speed

Apple M4 Pro GPU, `fill_u32` into device memory with no readback, minimum of 7, load 3 to 8:

| | GiB/s |
|---|---|
| Deno 2.9 (wgpu, Metal), 2^26 words, one fill per submit | 18.5 |
| Deno 2.9, 2^26 words, 32 fills per submit | 134 to 144 |
| Chromium (Dawn, Metal), 2^24 words, one fill per submit | 89 |
| Chromium, 2^26 and 2^27 words, one fill per submit | 167 |
| Chromium, 2^24 words, 16 fills per submit | 213 |
| store ceiling: one constant block per invocation, same buffer, 32 per submit | 700 to 720 |

A submit and its completion cost about 0.4 ms on wgpu, so a single small fill is latency
bound. The last row is the store ceiling of the same buffer and dispatch shape, and the
fill's own strided store pattern reaches it with constant data, so the fill is compute bound
at a fifth of the ceiling. Measured and rejected on this GPU, all within noise of the
committed shader or slower: a plain `*` in place of the exact 16-bit-half `mul_hi`, no
bounds check, workgroups of 64 or 128, four steps unrolled per store burst, a lane-major
thread mapping, two chunks per invocation, and the workgroup tile with 512-byte writes per
SIMD group from tandem-cuda, which measured four times slower (34 against 143 at K = 32).
Dawn's shader compiler gives 20 to 50% more than wgpu's on the same source. `deno task
bench` prints the first two rows and the ceiling.

`demo/index.html` runs the fill in a browser: `python3 -m http.server` in the repo root and
open `/demo/`.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
